"""Feature-level bandwidth impairment: uniform quantization of the BEV features that
collaborators share, applied via forward-pre-hooks so stock OpenCOOD models are not
modified. Assumes test-time batch_size=1 (row 0 of the agent-stacked tensor is ego;
rows 1..L-1 are collaborators — only those are quantized).

Interception points per model (verified against OpenCOOD commit 31ba160):
    PointPillarFCooper : fusion_net(spatial_features_2d, record_len)      rows (L,C,H,W)
    PointPillarV2VNet  : fusion_net(spatial_features_2d, record_len, ...) rows (L,C,H,W)
    PointPillarCoAlign : each fusion_net[i](feature_list[i], ...)         rows (L,C,H,W)
    CoBEVT (fax)       : fusion_net(regrouped, mask)                      (B,L,C,H,W)
    PointPillarIntermediate (AttFuse): fusion is interleaved inside the attention
        backbone, so quantization is applied to the backbone input ('spatial_features',
        the scatter output) — the earliest shared per-agent representation. Documented
        approximation: AttFuse has no single post-encoder message tensor.

Each hook also accumulates the message volume actually transmitted (numel x bits of
non-ego rows), giving bits/frame for free.
"""
import torch


def quantize(x, bits):
    """Uniform per-tensor quantization to 2^bits levels; float in, float out."""
    if bits >= 32:
        return x
    lo = x.min()
    hi = x.max()
    if torch.isclose(hi, lo):
        return x
    levels = float(2 ** bits - 1)
    q = torch.round((x - lo) / (hi - lo) * levels)
    return q / levels * (hi - lo) + lo


class BandwidthMeter:
    def __init__(self):
        self.bits_per_frame = []

    def add(self, numel, bits):
        self.bits_per_frame.append(numel * bits)

    @property
    def mean_bits(self):
        return (sum(self.bits_per_frame) / len(self.bits_per_frame)
                if self.bits_per_frame else 0.0)


def _quantize_rows(feat, bits, meter):
    """feat: (L, C, H, W); quantize rows 1..L-1 (collaborators)."""
    if feat.shape[0] <= 1:
        return feat
    out = feat.clone()
    out[1:] = quantize(feat[1:], bits)
    if meter is not None:
        meter.add(feat[1:].numel(), min(bits, 32))
    return out


def attach_bandwidth_hooks(model, bits, meter=None):
    """Install forward-pre-hooks on `model` quantizing collaborator features to
    `bits`. Returns a list of hook handles (call .remove() on each to detach).
    Raises KeyError for unsupported model classes."""
    name = type(model).__name__
    handles = []

    # InCoP (jorichee14/incop_analysis) shares OpenCOOD's agent-stacked message layout:
    # _run_fusion calls fusion_net(feature, record_len, affine_matrix[, ...]) with
    # feature of shape (L, C, H, W), so the same interception applies to all five of its
    # fusion methods — ours/CGRF, where2comm, cobevt, v2xvit, ermvp.
    if name in ('PointPillarFCooper', 'PointPillarV2VNet',
                'HeterModelBevfusionHighresIsaac'):
        def hook(module, args):
            feat = _quantize_rows(args[0], bits, meter)
            return (feat,) + tuple(args[1:])
        handles.append(model.fusion_net.register_forward_pre_hook(hook))

    elif name == 'PointPillarCoAlign':
        # multiscale: quantize each scale's input; count volume once per frame by
        # metering only the first scale.
        for i, mod in enumerate(model.fusion_net):
            def hook(module, args, _meter=(meter if i == 0 else None)):
                feat = _quantize_rows(args[0], bits, _meter)
                return (feat,) + tuple(args[1:])
            handles.append(mod.register_forward_pre_hook(hook))

    elif name in ('PointPillarCoBEVT', 'CorpBEVT', 'PointPillarFax'):
        def hook(module, args):
            regrouped = args[0]  # (B, L, C, H, W)
            if regrouped.shape[1] <= 1:
                return args
            out = regrouped.clone()
            out[:, 1:] = quantize(regrouped[:, 1:], bits)
            if meter is not None:
                meter.add(regrouped[:, 1:].numel(), min(bits, 32))
            return (out,) + tuple(args[1:])
        handles.append(model.fusion_net.register_forward_pre_hook(hook))

    elif name == 'PointPillarIntermediate':  # AttFuse: fusion inside backbone
        def hook(module, args):
            batch_dict = args[0]
            feat = batch_dict['spatial_features']
            batch_dict['spatial_features'] = _quantize_rows(feat, bits, meter)
            return args
        handles.append(model.backbone.register_forward_pre_hook(hook))

    else:
        raise KeyError('no bandwidth interception registered for model class %r'
                       % name)
    return handles
