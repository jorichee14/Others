# Baseline availability audit — can we actually run it?

**Checked 2026-08-21.** What can be added to the impairment matrix, at what cost.
The matrix currently holds seven architectures (late, early, attfuse, fcooper, v2vnet,
coalign, cobevt), all PointPillars, all from OpenCOOD's model zoo, all evaluated
pretrained.

| Method | Code | OPV2V config | OPV2V checkpoint | Verdict |
|---|---|---|---|---|
| **V2X-INCOP** | ❌ none published | — | — | **Cannot run.** Literature comparison only. |
| **Where2comm** | ✅ OpenCOOD-based | ❌ only `dair-v2x/` in `hypes_yaml/` | ❌ none found | Runnable **after** writing a config and training. |
| V2X-ViT | ✅ | ✅ | ✅ HEAL zoo (HuggingFace) | Free — verify spconv compat first. |
| DiscoNet | ✅ | ✅ | ✅ HEAL zoo (HuggingFace) | Free — verify spconv compat first. |
| V2VAM / LCRN | ❌ "code coming soon" | — | — | Cannot run. |
| TraF-Align | ✅ OpenCOOD-based | n/a — V2V4Real, V2X-Seq | ✅ released | Different dataset; the 400 ms latency bar. |

---

## V2X-INCOP — no code exists

Verified three ways: arXiv listing, the corresponding author's GitHub (7 public repos,
all paper lists or forks of mmdetection3d / vedaseg / dgl — no perception implementation),
and general search. No project page, no weights, no third-party reimplementation found.

**Do not reimplement it from the paper.** A reimplementation that underperforms proves
nothing — reviewers discount self-implemented competitors by default, and correctly so.

**Do this instead: match the metric.** INCOP's headline is *cooperative perception gain
over individual perception, averaged across packet drop rates, on OPV2V*. The ego-only
floor test **is** individual perception — same detector, same GT, collaborators withheld —
so the quantity is already computed for all seven methods. See
[`scripts/compare_incop_protocol.py`](../scripts/compare_incop_protocol.py).

### Result, from data already committed

Floor (individual perception) AP@0.7 = 0.575. Gain relative to floor, averaged over the
five drop-rate levels:

| Method | i.i.d. loss | bursty loss (GE) |
|---|---|---|
| cobevt | **+34.23%** | **+35.44%** |
| coalign | +26.61% | +28.03% |
| v2vnet | +24.66% | +26.23% |
| attfuse | +21.88% | +23.10% |
| late | +21.18% | +22.71% |
| fcooper | +20.66% | +22.12% |
| early | +20.42% | +21.84% |
| *V2X-INCOP (reported)* | *+14.06%* | — |

**Every unmodified baseline exceeds the figure INCOP reports for its loss-recovery
method.** Even plain late fusion, with no loss handling whatsoever, is at +21%.

**This is a protocol tension, not a refutation,** and must not be written as one until
four questions are answered from the paper's full text:

1. Is their "%" a *relative* gain or *absolute* mAP points? (14.06 points would read very
   differently — our best is +0.197 AP points.)
2. Which AP threshold? We have only AP@0.7 in the sweep summary.
3. Which drop rates, and is their "interruption" a **multi-frame outage** rather than
   per-frame loss? Sustained outage is our `loss_burst` axis at low `p_bg` — the
   `burst30_long` / `burst70_long` conditions built for the tracking phase — not
   `loss_iid`. The burst column above uses the standard GE settings, whose ~3.3-frame
   bursts are mild.
4. What detector is their "individual perception" baseline, and what absolute AP does it
   reach? A weaker single-agent baseline inflates relative gain mechanically.

*(arxiv.org is blocked by this environment's egress proxy; the full text has to be read
off-session.)*

### What it means for the direction either way

It is consistent with the parent study's own finding that **loss never crosses the
floor** on OPV2V while **latency crosses it at 100 ms for all seven methods**. If the
tension survives scrutiny, the reading is: *loss recovery addresses a failure mode that
does not bind on this dataset.* That strengthens the do-no-harm framing, which puts the
danger in staleness and misplaced trust rather than in missing packets.

---

## Where2comm — runnable, but it costs a training run

The public repo (`MediaBrain-SJTU/Where2comm`) is OpenCOOD-based and contains an
`opencood/` package, so `commchannel` attaches the same way it does to our existing
seven: instance monkeypatch of `retrieve_base_data`, plus a feature-hook entry for the
bandwidth axis.

Two obstacles, both real:

- **The released `hypes_yaml/` has only a `dair-v2x/` subdirectory.** OPV2V is claimed as
  supported in the README but is not configured in the release. The dataset/preprocess/
  postprocess blocks port from our existing OPV2V configs; the model block has to be
  written against their `where2comm` fusion module.
- **No OPV2V checkpoint anywhere** — not in their repo, not in CoAlign's zoo (which
  releases only CoAlign), not in HEAL's zoo (which does not include Where2comm). So it
  has to be trained: the ~100 GB OPV2V train split on a 12 GB card.

**It is still worth it, and it is the single highest-value addition.** Where2comm's
spatial confidence map is co-trained with the detection head. It is the canonical
*uncertainty-from-the-model* system, and therefore the sharpest possible target for the
calibration claim: **the method whose entire design premise is knowing where information
is valuable turns out to be miscalibrated with respect to how that information arrived.**
Its inference-time bandwidth knob also pairs directly with our bandwidth-quantization
axis, giving a second, independent handle on the same trade-off.

Sequence it **after** the calibration measurement lands on the existing seven. If the
effect does not exist on checkpoints already on disk, a training run will not create it.

---

## The cheap win first: V2X-ViT and DiscoNet

HEAL's HuggingFace zoo (`yifanlu/HEAL`) releases OPV2V checkpoints for AttFuse, Cooper,
F-Cooper, V2VNet, **DiscoNet**, **V2X-ViT**, CoAlign, and HEAL. Two of those are not in
our matrix, and one of them matters a lot:

**V2X-ViT is the canonical "compensate implicitly by training under simulated delay"
system** — it handles asynchrony by having seen it in training rather than by modelling
time. That makes it the ideal foil: if a model trained *on* delay is still miscalibrated
*with respect to* delay, the argument that implicit robustness is not the same as knowing
what to trust becomes very hard to dispute.

**Blocking check before counting on this:** HEAL states its checkpoints are stored under
**spconv 1.2.1 and are not compatible with spconv 2.x**, and this environment runs
`spconv-cu117 2.3.6` — the same install that already cost a debugging cycle over the dual
`cumm` shadowing. Verify one checkpoint loads before planning around the other seven.
