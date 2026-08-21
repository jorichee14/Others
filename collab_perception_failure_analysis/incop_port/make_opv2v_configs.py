#!/usr/bin/env python
"""Generate OPV2V configs for the InCoP codebase (jorichee14/incop_analysis).

WHY A GENERATOR AND NOT FOUR YAMLs. The four configs share ~90% of their content —
dataset paths, ranges, voxelisation, encoder, backbone, anchors, loss, optimiser. Only
the fusion block differs. Hand-maintaining four near-identical 300-line files is how
they silently drift apart, and a drifted config is an unattributable result.

WHAT THIS PORTS. InCoP's `heter_model_bevfusion_highres_isaac` is only nominally
Isaac-specific: its single Isaac dependency is the encoder lookup, and
`_find_encoder_class_isaac` (heter_encoders_isaac.py:94) falls back to the generic
`_find_encoder_class`. So `core_method: point_pillar` resolves to the plain PointPillar
encoder and the model runs on ordinary OPV2V data with no code change. `build_dataset`
already accepts `dataset: opv2v`, and modality assignment is optional, so a homogeneous
LiDAR-only OPV2V run needs configs and nothing else.

HEAD TYPE. Default `anchor_based`, matching the seven OpenCOOD baselines in the parent
study: it emits cls_preds/reg_preds/dir_preds, which VoxelPostprocessor and the parent
study's verified `compute_metrics` consume unchanged. That requires `point_pillar_loss.py`
which InCoP does not ship — see install_overlay.sh. Pass --head center_head to use the
loss InCoP already has, at the cost of a different detection head from the baselines.

Usage:
    python make_opv2v_configs.py --opv2v-root ~/cpfa/data/OPV2V --out <incop>/opencood/hypes_yaml/opv2v
"""
import argparse
import os

import yaml

# OPV2V LiDAR-only PointPillars conventions, matching the parent study's seven baselines.
RANGE = [-140.8, -40, -3, 140.8, 40, 1]
VOXEL = [0.4, 0.4, 4]
MAX_CAV = 5
COMM_RANGE = 70

DIR_ARGS = {'dir_offset': 0.7853, 'num_bins': 2, 'anchor_yaw': [0, 90]}

# Fusion blocks. agent_size must equal max_cav — the Isaac configs use 2 because the
# indoor benchmark is a two-robot setup; OPV2V has up to five agents per scene.
FUSION = {
    'ours': {
        'feat_dim': 64, 'swin_window_size': 4, 'swin_shift_size': 2,
        'swin_num_heads': 4, 'swin_drop': 0.0,
        'residual_output_init_gain': 0.01,
        'use_density_quality': True, 'require_support_cue': True,
        'density_sparse_communication': True, 'communication_density_threshold': 0.0,
        'communication_dtype_bytes': 4, 'communication_index_bytes': 4,
        'communication_metadata_dtype_bytes': 4,
    },
    'where2comm': {
        'input_dim': 64, 'model_dim': 256, 'spatial_downsample_stages': 0,
        'heads': 8, 'agent_size': MAX_CAV, 'threshold': 0.01,
        'communication_rounds': 1, 'gaussian_smooth': True,
        'gaussian_kernel_size': 5, 'gaussian_sigma': 1.0,
    },
    'cobevt': {
        'input_dim': 64, 'model_dim': 256, 'spatial_downsample_stages': 0,
        'mlp_dim': 1024, 'agent_size': MAX_CAV, 'window_size': 4,
        'dim_head': 32, 'drop_out': 0.1, 'depth': 3,
    },
    'v2xvit': {
        'input_dim': 64, 'model_dim': 256, 'spatial_downsample_stages': 0,
        'agent_size': MAX_CAV,
    },
    'ermvp': {
        'input_dim': 64, 'model_dim': 256, 'spatial_downsample_stages': 0,
        'agent_size': MAX_CAV,
    },
}

# CGRF decodes after fusion; the attention-style fusions reuse the shared head path.
DECODER = {
    'decoder_args': {
        'layer_nums': [3, 5, 8], 'layer_strides': [1, 2, 2],
        'num_filters': [64, 128, 256], 'upsample_strides': [1, 2, 4],
        'num_upsample_filter': [128, 128, 128],
    },
    'decoder_shrink_header': {
        'kernal_size': [3], 'stride': [1], 'padding': [1],
        'dim': [256], 'input_dim': 384,
    },
}


def voxel_preprocess(train_max=32000, test_max=70000, pts=32):
    return {
        'core_method': 'SpVoxelPreprocessor',
        'args': {'voxel_size': VOXEL, 'max_points_per_voxel': pts,
                 'max_voxel_train': train_max, 'max_voxel_test': test_max},
        'cav_lidar_range': RANGE,
    }


def encoder_block():
    """LiDAR-only PointPillar branch. `_find_encoder_class_isaac` falls through to this."""
    return {
        'core_method': 'point_pillar',
        'sensor_type': 'lidar',
        'encoder_args': {
            'voxel_size': VOXEL,
            'lidar_range': RANGE,
            'pillar_vfe': {'use_norm': True, 'with_distance': False,
                           'use_absolute_xyz': True, 'num_filters': [64]},
            'point_pillar_scatter': {'num_features': 64},
        },
        # Pre-fusion message backbone: features are strided by 2 before transmission,
        # so the anchor grid sits at feature_stride 2 exactly as in OpenCOOD.
        'backbone_args': {'layer_nums': [3], 'layer_strides': [2], 'num_filters': [64]},
        'aligner_args': {'core_method': 'identity'},
        'layers_args': {
            'layer_nums': [3, 5, 8], 'layer_strides': [1, 2, 2],
            'num_filters': [64, 128, 256], 'upsample_strides': [1, 2, 4],
            'num_upsample_filter': [128, 128, 128],
        },
        'shrink_header': {'kernal_size': [3], 'stride': [1], 'padding': [1],
                          'dim': [256], 'input_dim': 384},
    }


def base(name, root, head, epochs, batch):
    return {
        'name': name,
        'root_dir': os.path.join(root, 'train'),
        'validate_dir': os.path.join(root, 'validate'),
        'test_dir': os.path.join(root, 'test'),
        'yaml_parser': 'load_general_params',
        'train_params': {'batch_size': batch, 'epoches': epochs,
                         'eval_freq': 2, 'save_freq': 2, 'max_cav': MAX_CAV},
        'comm_range': COMM_RANGE,
        'input_source': ['lidar'],
        'label_type': 'lidar',
        'cav_lidar_range': RANGE,
        'add_data_extension': [],
        'data_augment': [
            {'NAME': 'random_world_flip', 'ALONG_AXIS_LIST': ['x']},
            {'NAME': 'random_world_rotation',
             'WORLD_ROT_ANGLE': [-0.78539816, 0.78539816]},
            {'NAME': 'random_world_scaling', 'WORLD_SCALE_RANGE': [0.95, 1.05]},
        ],
        # Top-level preprocess is a placeholder; the real voxelisation is per-modality,
        # exactly as in the Isaac configs.
        'preprocess': voxel_preprocess(train_max=1, test_max=1, pts=1),
        'postprocess': postprocess(head),
        'loss': loss(head),
        'optimizer': {'core_method': 'AdamW', 'lr': 0.002,
                      'args': {'eps': 1.0e-08, 'weight_decay': 0.01}},
        'lr_scheduler': {'core_method': 'multistep', 'gamma': 0.1,
                         'step_size': [15, 25]},
    }


def postprocess(head):
    p = {
        'core_method': 'VoxelPostprocessor',
        'gt_range': RANGE,
        'anchor_args': {
            'cav_lidar_range': RANGE,
            'l': 3.9, 'w': 1.6, 'h': 1.56, 'r': [0, 90],
            'feature_stride': 2, 'num': 2,
        },
        'target_args': {'pos_threshold': 0.6, 'neg_threshold': 0.45,
                        'score_threshold': 0.20},
        'order': 'hwl',
        'max_num': 100,
        'nms_thresh': 0.15,
    }
    if head == 'anchor_based':
        p['dir_args'] = dict(DIR_ARGS)
    return p


def loss(head):
    if head == 'center_head':
        return {'core_method': 'center_head_loss',
                'args': {'lidar_range': RANGE, 'class_names': ['vehicle'],
                         'num_classes': 1, 'num_max_objs': 100,
                         'gaussian_overlap': 0.1, 'min_radius': 2,
                         'cls_weight': 1.0, 'loc_weight': 2.0,
                         'code_weights': [1.0] * 8}}
    # HEAL/CoAlign schema — must match the point_pillar_loss.py copied by
    # install_overlay.sh. If you copy DerrickXuNu/OpenCOOD's simpler variant instead,
    # replace this block with that file's expected keys.
    return {
        'core_method': 'point_pillar_loss',
        'args': {
            'pos_cls_weight': 2.0,
            'cls': {'type': 'SigmoidFocalLoss', 'alpha': 0.25,
                    'gamma': 2.0, 'weight': 1.0},
            'reg': {'type': 'WeightedSmoothL1Loss', 'sigma': 3.0,
                    'codewise': True, 'weight': 2.0},
            'dir': {'type': 'WeightedSoftmaxClassificationLoss', 'weight': 0.2,
                    'args': dict(DIR_ARGS)},
        },
    }


def head_args(head):
    if head == 'center_head':
        return {'head_type': 'center_head', 'in_head': 256,
                'center_head': {'class_names': ['vehicle'], 'num_classes': 1,
                                'shared_conv_channels': 128, 'fuse_final_conv': False,
                                'use_bias_before_norm': True, 'num_hm_conv': 2,
                                'init_bias': -2.19,
                                'separate_head': {
                                    'num_conv': 2, 'head_conv_channels': 64,
                                    'head_dict': {
                                        'center': {'out_channels': 2, 'num_conv': 2},
                                        'center_z': {'out_channels': 1, 'num_conv': 2},
                                        'dim': {'out_channels': 3, 'num_conv': 2},
                                        'rot': {'out_channels': 2, 'num_conv': 2}}}}}
    return {'head_type': 'anchor_based', 'in_head': 256,
            'anchor_number': 2, 'dir_args': dict(DIR_ARGS)}


def single_config(root, head, epochs, batch):
    """Single-agent pretraining. Produces the encoder_m1 weights the fusion configs load."""
    cfg = base('opv2v_single_m1_pointpillar_lidar', root, head, epochs, batch)
    cfg['fusion'] = {'core_method': 'intermediateheter', 'dataset': 'opv2v',
                     'args': {'proj_first': False, 'grid_conf': 'None',
                              'data_aug_conf': 'None'}}
    cfg['train_params']['max_cav'] = 1
    cfg['heter'] = heter_block()
    m = {'ego_modality': 'm1', 'lidar_range': RANGE,
         'supervise_single': False, 'single_head_shared': False,
         'pre_fusion_message_backbone': True,
         'm1': encoder_block(),
         'fusion_method': 'cobevt', 'cobevt': dict(FUSION['cobevt'])}
    m.update(head_args(head))
    cfg['model'] = {'core_method': 'heter_model_bevfusion_highres_isaac', 'args': m}
    return cfg


def heter_block():
    return {
        'assignment_path': None,          # homogeneous: every CAV is m1
        'ego_modality': 'm1',
        'mapping_dict': {'m1': 'm1'},
        'modality_setting': {
            'm1': {'sensor_type': 'lidar', 'core_method': 'point_pillar',
                   'preprocess': voxel_preprocess()},
        },
    }


def fusion_config(method, root, head, epochs, batch, pretrained):
    cfg = base('opv2v_%s_lidar' % method, root, head, epochs, batch)
    cfg['fusion'] = {'core_method': 'intermediateheter', 'dataset': 'opv2v',
                     'args': {'proj_first': False, 'grid_conf': 'None',
                              'data_aug_conf': 'None'}}
    cfg['heter'] = heter_block()
    if pretrained:
        cfg['isaac_pretrained'] = {'enabled': True, 'path': pretrained,
                                   'checkpoint_mode': 'bestval',
                                   'load_prefixes': ['encoder_m1']}
    m = {'ego_modality': 'm1', 'lidar_range': RANGE,
         'supervise_single': False, 'single_head_shared': False,
         'pre_fusion_message_backbone': True,
         'm1': encoder_block(),
         'fusion_method': method, method: dict(FUSION[method])}
    if method == 'ours':
        m.update({k: dict(v) for k, v in DECODER.items()})
    m.update(head_args(head))
    cfg['model'] = {'core_method': 'heter_model_bevfusion_highres_isaac', 'args': m}
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--opv2v-root', required=True,
                    help='directory containing train/ validate/ test/')
    ap.add_argument('--out', required=True, help='<incop>/opencood/hypes_yaml/opv2v')
    ap.add_argument('--head', default='anchor_based',
                    choices=['anchor_based', 'center_head'])
    ap.add_argument('--methods', default='ours,where2comm,cobevt',
                    help='comma separated; cobevt is the cross-codebase bridge')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=2,
                    help='2 fits a 12 GB card at this range; raise if you have headroom')
    ap.add_argument('--pretrained', default='',
                    help='single-agent run dir, e.g. opencood/logs/opv2v_single_...')
    args = ap.parse_args()

    root = os.path.expanduser(args.opv2v_root)
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)

    written = [('single_m1_pointpillar.yaml',
                single_config(root, args.head, args.epochs, args.batch_size))]
    for method in [m.strip() for m in args.methods.split(',') if m.strip()]:
        if method not in FUSION:
            raise SystemExit('unknown method %r (have: %s)'
                             % (method, ', '.join(sorted(FUSION))))
        written.append(('%s.yaml' % method,
                        fusion_config(method, root, args.head, args.epochs,
                                      args.batch_size, args.pretrained)))

    for fname, cfg in written:
        path = os.path.join(out, fname)
        with open(path, 'w') as f:
            yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
        print('wrote %s' % path)

    if not args.pretrained:
        print('\nNOTE: --pretrained was empty, so the fusion configs train their encoder')
        print('from scratch. Train single_m1_pointpillar.yaml first, then regenerate with')
        print('--pretrained opencood/logs/<that run dir> to match the InCoP recipe.')


if __name__ == '__main__':
    main()
