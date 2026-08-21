#!/usr/bin/env python
"""Generate LiDAR-only configs for the InCoP codebase (jorichee14/incop_analysis).

Two presets, because the study needs two arms and they must not drift apart:

  --preset opv2v   outdoor V2V, 281.6 x 80 m, 0.4 m voxels, 1 vehicle class, up to 5
                   agents, anchor-based head. Comparable to the seven OpenCOOD baselines.
                   Expensive: needs the OPV2V train+validate splits and ~12-20 h/model.

  --preset incop   InCoP's own indoor benchmark, 22.4 x 22.4 m, 0.1 m voxels, 7 indoor
                   classes, 2 robots, center head. Cheap: InCoP's own single-agent LiDAR
                   config already trains at batch_size 8 for 25 epochs, so this is a
                   different cost class entirely -- roughly an hour or three per model.

BOTH ARMS ARE LiDAR-ONLY, DELIBERATELY. InCoP's indoor configs are multimodal
(BEVFusion + a DINOv3 camera branch), but running cameras indoors and not outdoors would
confound the thing the pairing exists to measure: the indoor/outdoor difference would mix
scene dynamics with a modality change. LiDAR-only on both sides isolates dynamics. It
also drops the DINOv3 forward pass, which is what makes the indoor arm cheap.

The tradeoff is stated rather than hidden: CGRF is Complementarity-Guided, and LiDAR-only
may remove some of the complementarity it exploits, so the LiDAR-only result risks
under-representing it. Run the multimodal arm afterwards as a second measurement -- but
compare it to the LiDAR-only indoor arm, not across benchmarks.

WHY THIS PORTS AT ALL. `heter_model_bevfusion_highres_isaac` is only nominally
Isaac-specific: its one Isaac dependency is the encoder lookup, and
`_find_encoder_class_isaac` (heter_encoders_isaac.py:94) falls back to the generic
`_find_encoder_class`, so `core_method: point_pillar` resolves to the plain PointPillar
encoder. `build_dataset` already accepts both `opv2v` and `isaacsim`. The model branches
on `sensor_type != "camera"`, so `lidar` takes the LiDAR path. No model code changes.

Usage:
    python make_configs.py --preset incop --data-root ~/InCoP/dataset --scene hospital \
        --out <incop>/opencood/hypes_yaml/incop_lidar
    python make_configs.py --preset opv2v --data-root ~/cpfa/data/OPV2V \
        --out <incop>/opencood/hypes_yaml/opv2v
"""
import argparse
import os

import yaml

DIR_ARGS = {'dir_offset': 0.7853, 'num_bins': 2, 'anchor_yaw': [0, 90]}

INDOOR_CLASSES = ['potted_plant', 'chair', 'medical_bag', 'traffic_cone',
                  'wet_floor_sign', 'fire_extinguisher', 'trash_can']

PRESETS = {
    'opv2v': {
        'dataset': 'opv2v',
        'range': [-140.8, -40, -3, 140.8, 40, 1],
        'voxel': [0.4, 0.4, 4],
        'max_cav': 5,
        'comm_range': 70,
        'head': 'anchor_based',
        'classes': ['vehicle'],
        'anchor': {'l': 3.9, 'w': 1.6, 'h': 1.56},
        'epochs': 30,
        'batch_size': 2,          # 12 GB card at 704x200
        'lr_steps': [15, 25],
        # layer_strides [1,2,2] after a stride-2 pre-fusion backbone -> feature_stride 2
        'layer_strides': [1, 2, 2],
        'voxel_train': 32000, 'voxel_test': 70000,
        'target': {'pos_threshold': 0.6, 'neg_threshold': 0.45, 'score_threshold': 0.20},
        'nms': {'max_num': 100, 'nms_thresh': 0.15},
        'fixed_order': False,
        'splits': ('train', 'validate', 'test'),
    },
    'incop': {
        'dataset': 'isaacsim',
        'range': [0.0, -11.2, -1, 22.4, 11.2, 3],
        'voxel': [0.1, 0.1, 4],
        'max_cav': 2,
        'comm_range': 50,
        'head': 'center_head',    # InCoP ships center_head_loss; no loss port needed
        'classes': INDOOR_CLASSES,
        'anchor': {'l': 1.0, 'w': 1.0, 'h': 1.0},
        'epochs': 25,             # InCoP's own recipe
        'batch_size': 8,          # InCoP's own single-agent LiDAR config uses 8
        'lr_steps': [15, 30],
        'layer_strides': [1, 2, 2],
        'voxel_train': 60000, 'voxel_test': 120000,
        'target': {'pos_threshold': 0.35, 'neg_threshold': 0.2, 'score_threshold': 0.25},
        'nms': {'max_num': 1024, 'pre_nms_topk': 4096, 'post_nms_topk': 512,
                'nms_thresh': 0.15},
        'fixed_order': True,      # 2 robots, fixed ego
        'splits': ('train', 'validate', 'test'),
    },
}

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
        'heads': 8, 'threshold': 0.01, 'communication_rounds': 1,
        'gaussian_smooth': True, 'gaussian_kernel_size': 5, 'gaussian_sigma': 1.0,
    },
    'cobevt': {
        'input_dim': 64, 'model_dim': 256, 'spatial_downsample_stages': 0,
        'mlp_dim': 1024, 'window_size': 4, 'dim_head': 32, 'drop_out': 0.1, 'depth': 3,
    },
    'v2xvit': {'input_dim': 64, 'model_dim': 256, 'spatial_downsample_stages': 0},
    'ermvp': {'input_dim': 64, 'model_dim': 256, 'spatial_downsample_stages': 0},
}
NEEDS_AGENT_SIZE = ('where2comm', 'cobevt', 'v2xvit', 'ermvp')

DECODER = {
    'decoder_args': {'layer_nums': [3, 5, 8], 'layer_strides': [1, 2, 2],
                     'num_filters': [64, 128, 256], 'upsample_strides': [1, 2, 4],
                     'num_upsample_filter': [128, 128, 128]},
    'decoder_shrink_header': {'kernal_size': [3], 'stride': [1], 'padding': [1],
                              'dim': [256], 'input_dim': 384},
}

# DCG lives at model.args.lidar_support_mask, NOT in the fusion block: the model does
# encoder_args.setdefault("lidar_support_mask", args.get(..., {})), so omitting it gives
# {} -> gate off -> the DENSE CLC ABLATION mislabelled as CGRF. Only `ours` uses it.
# q95 is DATASET-SPECIFIC: 3.7377 (= log 15) was estimated on 512 indoor hospital sweeps.
DCG_HOSPITAL_Q95 = 3.737669618283368


def dcg_block(q95):
    return {'enabled': True, 'mode': 'log_density', 'log_density_q95': q95,
            'dilation_radius': 0, 'apply_to_feature': True,
            'apply_stage': 'pre_cooperative_fusion'}


def voxel_preprocess(p, train_max=None, test_max=None, pts=32):
    return {'core_method': 'SpVoxelPreprocessor',
            'args': {'voxel_size': p['voxel'], 'max_points_per_voxel': pts,
                     'max_voxel_train': train_max or p['voxel_train'],
                     'max_voxel_test': test_max or p['voxel_test']},
            'cav_lidar_range': p['range']}


def encoder_block(p):
    """LiDAR-only PointPillar branch; the Isaac encoder lookup falls through to this."""
    return {
        'core_method': 'point_pillar', 'sensor_type': 'lidar',
        'encoder_args': {
            'voxel_size': p['voxel'], 'lidar_range': p['range'],
            'pillar_vfe': {'use_norm': True, 'with_distance': False,
                           'use_absolute_xyz': True, 'num_filters': [64]},
            'point_pillar_scatter': {'num_features': 64}},
        'backbone_args': {'layer_nums': [3], 'layer_strides': [2],
                          'num_filters': [64]},
        'aligner_args': {'core_method': 'identity'},
        'layers_args': {'layer_nums': [3, 5, 8], 'layer_strides': p['layer_strides'],
                        'num_filters': [64, 128, 256], 'upsample_strides': [1, 2, 4],
                        'num_upsample_filter': [128, 128, 128]},
        'shrink_header': {'kernal_size': [3], 'stride': [1], 'padding': [1],
                          'dim': [256], 'input_dim': 384},
    }


def postprocess(p):
    out = {'core_method': 'VoxelPostprocessor', 'gt_range': p['range'],
           'anchor_args': dict(cav_lidar_range=p['range'], r=[0, 90],
                               feature_stride=2, num=2, **p['anchor']),
           'target_args': dict(p['target']), 'order': 'hwl'}
    out.update(p['nms'])
    if p['head'] == 'anchor_based':
        out['dir_args'] = dict(DIR_ARGS)
    else:
        out['class_names'] = list(p['classes'])
    return out


def loss(p):
    if p['head'] == 'center_head':
        return {'core_method': 'center_head_loss',
                'args': {'lidar_range': p['range'], 'class_names': list(p['classes']),
                         'num_classes': len(p['classes']),
                         'num_max_objs': p['nms']['max_num'],
                         'gaussian_overlap': 0.1, 'min_radius': 2,
                         'cls_weight': 1.0, 'loc_weight': 2.0,
                         'code_weights': [1.0] * 8}}
    # HEAL/CoAlign schema -- must match the point_pillar_loss.py install_overlay.sh copies.
    return {'core_method': 'point_pillar_loss',
            'args': {'pos_cls_weight': 2.0,
                     'cls': {'type': 'SigmoidFocalLoss', 'alpha': 0.25,
                             'gamma': 2.0, 'weight': 1.0},
                     'reg': {'type': 'WeightedSmoothL1Loss', 'sigma': 3.0,
                             'codewise': True, 'weight': 2.0},
                     'dir': {'type': 'WeightedSoftmaxClassificationLoss', 'weight': 0.2,
                             'args': dict(DIR_ARGS)}}}


def head_args(p):
    if p['head'] == 'anchor_based':
        return {'head_type': 'anchor_based', 'in_head': 256,
                'anchor_number': 2, 'dir_args': dict(DIR_ARGS)}
    return {'head_type': 'center_head', 'in_head': 256,
            'min_size': 0.05, 'max_size': 6.0,
            'center_head': {
                'class_names': list(p['classes']), 'num_classes': len(p['classes']),
                'shared_conv_channels': 128, 'fuse_final_conv': False,
                'use_bias_before_norm': True, 'num_hm_conv': 2, 'init_bias': -2.19,
                'separate_head': {'num_conv': 2, 'head_conv_channels': 64,
                                  'head_dict': {
                                      'center': {'out_channels': 2, 'num_conv': 2},
                                      'center_z': {'out_channels': 1, 'num_conv': 2},
                                      'dim': {'out_channels': 3, 'num_conv': 2},
                                      'rot': {'out_channels': 2, 'num_conv': 2}}}}}


def base(name, p, root, assignment):
    tr, va, te = p['splits']
    cfg = {
        'name': name,
        'root_dir': os.path.join(root, tr),
        'validate_dir': os.path.join(root, va),
        'test_dir': os.path.join(root, te),
        'yaml_parser': 'load_general_params',
        'train_params': {'batch_size': p['batch_size'], 'epoches': p['epochs'],
                         'eval_freq': 2, 'save_freq': 2, 'max_cav': p['max_cav']},
        'comm_range': p['comm_range'],
        'input_source': ['lidar'],
        'label_type': 'lidar',
        'cav_lidar_range': p['range'],
        'add_data_extension': [],
        'fusion': {'core_method': 'intermediateheter', 'dataset': p['dataset'],
                   'args': {'proj_first': False, 'grid_conf': 'None',
                            'data_aug_conf': 'None'}},
        'data_augment': [
            {'NAME': 'random_world_flip', 'ALONG_AXIS_LIST': ['x']},
            {'NAME': 'random_world_rotation',
             'WORLD_ROT_ANGLE': [-0.39269908, 0.39269908]},
            {'NAME': 'random_world_scaling', 'WORLD_SCALE_RANGE': [0.95, 1.05]}],
        # Top-level preprocess is a placeholder; real voxelisation is per-modality.
        'preprocess': voxel_preprocess(p, train_max=1, test_max=1, pts=1),
        'heter': {'assignment_path': assignment, 'ego_modality': 'm1',
                  'mapping_dict': {'m1': 'm1', 'm2': 'm1'},
                  'modality_setting': {'m1': {'sensor_type': 'lidar',
                                              'core_method': 'point_pillar',
                                              'preprocess': voxel_preprocess(p)}}},
        'postprocess': postprocess(p),
        'loss': loss(p),
        'optimizer': {'core_method': 'AdamW', 'lr': 0.002 if p['dataset'] == 'opv2v'
                      else 0.001, 'args': {'eps': 1.0e-08, 'weight_decay': 0.01}},
        'lr_scheduler': {'core_method': 'multistep', 'gamma': 0.1,
                         'step_size': p['lr_steps']},
    }
    if p['fixed_order']:
        cfg['train_params'].update({'fixed_cav_order': True, 'fixed_ego_id': 0,
                                    'fixed_cav_ids_order': [0, 1]})
    return cfg


def build(method, p, root, assignment, pretrained, q95, single=False):
    name = '%s_%s_lidar' % (p['dataset'], 'single_m1_pointpillar' if single else method)
    cfg = base(name, p, root, assignment)
    if single:
        cfg['train_params']['max_cav'] = 1
    elif pretrained:
        cfg['isaac_pretrained'] = {'enabled': True, 'path': pretrained,
                                   'checkpoint_mode': 'bestval',
                                   'load_prefixes': ['encoder_m1']}
    fusion = dict(FUSION[method])
    if method in NEEDS_AGENT_SIZE:
        fusion['agent_size'] = p['max_cav']
    m = {'ego_modality': 'm1', 'lidar_range': p['range'],
         'supervise_single': False, 'single_head_shared': False,
         'pre_fusion_message_backbone': True,
         'm1': encoder_block(p), 'fusion_method': method, method: fusion}
    if method == 'ours':
        m.update({k: dict(v) for k, v in DECODER.items()})
        m['lidar_support_mask'] = dcg_block(q95)
    m.update(head_args(p))
    cfg['model'] = {'core_method': 'heter_model_bevfusion_highres_isaac', 'args': m}
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', required=True, choices=sorted(PRESETS))
    ap.add_argument('--data-root', required=True,
                    help='opv2v: dir containing train/validate/test. '
                         'incop: the dataset/ dir containing IsaacSimOPV2V_<scene>/')
    ap.add_argument('--scene', default='hospital',
                    help='incop only: hospital | office | warehouse')
    ap.add_argument('--out', required=True, help='output hypes_yaml directory')
    ap.add_argument('--methods', default='ours,where2comm,cobevt',
                    help='comma separated; cobevt is the cross-codebase bridge')
    ap.add_argument('--pretrained', default='', help='single-agent run dir')
    ap.add_argument('--dcg-q95', type=float, default=None,
                    help="95th percentile of log(1+D) over this dataset's train sweeps")
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch-size', type=int, default=None)
    args = ap.parse_args()

    p = dict(PRESETS[args.preset])
    if args.epochs:
        p['epochs'] = args.epochs
    if args.batch_size:
        p['batch_size'] = args.batch_size

    root = os.path.expanduser(args.data_root)
    assignment = None
    if args.preset == 'incop':
        root = os.path.join(root, 'IsaacSimOPV2V_%s' % args.scene)
        assignment = os.path.join(root, 'heter_modality_assign.json')

    q95 = args.dcg_q95 if args.dcg_q95 is not None else DCG_HOSPITAL_Q95
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)

    written = [('single_m1_pointpillar.yaml',
                build('cobevt', p, root, assignment, '', q95, single=True))]
    for method in [m.strip() for m in args.methods.split(',') if m.strip()]:
        if method not in FUSION:
            raise SystemExit('unknown method %r (have: %s)'
                             % (method, ', '.join(sorted(FUSION))))
        written.append(('%s.yaml' % method,
                        build(method, p, root, assignment, args.pretrained, q95)))

    for fname, cfg in written:
        with open(os.path.join(out, fname), 'w') as f:
            yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
        print('wrote %s' % os.path.join(out, fname))

    if 'ours' in args.methods and args.dcg_q95 is None and args.preset != 'incop':
        print("\nWARNING: CGRF's density gate is using InCoP's INDOOR q95 = %.4f"
              % DCG_HOSPITAL_Q95)
        print('(= log 15, estimated on 512 hospital training sweeps). This dataset has a')
        print('different LiDAR density distribution, so the gate is miscalibrated until')
        print('you pass --dcg-q95 <95th pct of log(1+D) over its train sweeps>.')
        print('Too low: everything passes, CGRF degenerates toward dense CLC.')
        print('Too high: the mask starves the fusion of partner evidence.')
    if args.preset == 'incop' and args.dcg_q95 is None and args.scene != 'hospital':
        print('\nNOTE: q95 = %.4f was estimated on the HOSPITAL scene. Re-estimate for %s.'
              % (DCG_HOSPITAL_Q95, args.scene))
    if not args.pretrained:
        print('\nNOTE: --pretrained empty, so fusion configs train the encoder from')
        print('scratch. Train single_m1_pointpillar.yaml first, then regenerate with')
        print('--pretrained opencood/logs/<run dir> to match the InCoP recipe.')


if __name__ == '__main__':
    main()
