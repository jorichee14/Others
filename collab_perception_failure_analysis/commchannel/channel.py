"""CommChannel: dataset-level impairment wrapper for OpenCOOD.

Attaches to a built OpenCOOD dataset by replacing its bound `retrieve_base_data` with
an impaired version. Stock OpenCOOD is never modified on disk. The impaired retrieval
mirrors the stock logic at commit 31ba160 (scenario/timestamp lookup, `reform_param`
for parameter assembly) and inherits its key semantic: ground-truth annotations always
come from the CURRENT timestamp regardless of message delay.

GT policy under impairment: the wrapped dataset's own GT can shrink when collaborators
are dropped (OpenCOOD builds GT as a union over the CAVs in the sample). Evaluation
runners must therefore take predictions from the impaired dataset but ground truth from
a parallel clean dataset (same pattern as the Phase 1 `nocomm` mode). See
`scripts/run_phase2_identity.py` for the reference implementation.

Impairments applied here (data level, works for late/early/intermediate fusion):
    drop (packet loss)  - collaborator removed from the sample entirely
    blockage            - collaborator removed when a labeled vehicle stands on the
                          ego<->collaborator chord (scene-conditioned loss; see
                          blockage.py and docs/BLOCKAGE.md)
    latency / staleness - collaborator lidar + reported pose come from a past frame
    pose noise          - collaborator's cav->ego transformation recomputed from a
                          perturbed pose (gt_transformation_matrix left untouched)
    ghost vehicles      - car-shaped point clusters appended to collaborator's cloud
    scene swap          - collaborator's cloud replaced with one from another scenario

Bandwidth quantization is feature-level: see feature_hooks.py.
"""
import math
from collections import OrderedDict

import numpy as np

from .blockage import BlockageTable, locate_frame
from .schedule import Schedule

# Ghost vehicle dimensions (m): typical sedan, matching OPV2V vehicle scale.
GHOST_LWH = (4.5, 1.9, 1.6)
GHOST_POINTS = 220           # points per ghost, roughly a mid-range vehicle return
GHOST_RANGE_XY = (8.0, 60.0)  # ghosts placed in this radial band from the collaborator


class CommChannel:
    def __init__(self, config, blockage=None):
        self.cfg = config
        self.blockage = blockage
        self.schedule = Schedule(config, blockage)
        self._attached = False
        self.reset_stats()

    # ------------------------------------------------------------------- stats
    def reset_stats(self):
        """Delivery accounting. `blockage_p` sweeps have a data-determined loss
        rate, and the matched-PDR comparison is the whole point of that family —
        so the realized rate is measured here rather than inferred from config."""
        self.stats = {'messages': 0, 'dropped_channel': 0, 'dropped_empty': 0,
                      'blocked': 0}

    def stats_dict(self):
        n = self.stats['messages']
        d = dict(self.stats)
        dropped = self.stats['dropped_channel'] + self.stats['dropped_empty']
        d['realized_drop_rate'] = round(dropped / n, 5) if n else None
        d['blocked_rate'] = round(self.stats['blocked'] / n, 5) if n else None
        return d

    # ------------------------------------------------------------------ attach
    def attach(self, dataset):
        """Monkeypatch dataset.retrieve_base_data (instance-level) with the impaired
        version. Returns the dataset for chaining."""
        if self._attached:
            raise RuntimeError('channel already attached')
        if self.cfg.uses_blockage and self.blockage is None:
            # Build on the standard grid plus whatever this config asks for, so a
            # single cached table serves every clearance the study uses.
            from .blockage import DEFAULT_CLEARANCES
            grid = tuple(sorted(set(DEFAULT_CLEARANCES)
                                | {float(self.cfg.blockage_clearance)}))
            self.blockage = BlockageTable.load_or_build(dataset, clearances=grid)
            self.schedule.blockage = self.blockage
        self._dataset = dataset
        self._orig_retrieve = dataset.retrieve_base_data
        channel = self

        def impaired_retrieve(idx, cur_ego_pose_flag=True):
            return channel._retrieve(dataset, idx, cur_ego_pose_flag)

        dataset.retrieve_base_data = impaired_retrieve
        self._attached = True
        return dataset

    def detach(self):
        if self._attached:
            self._dataset.retrieve_base_data = self._orig_retrieve
            self._attached = False

    # ------------------------------------------------------------ core retrieve
    def _locate(self, dataset, idx):
        return locate_frame(dataset, idx)

    def _retrieve(self, dataset, idx, cur_ego_pose_flag):
        from opencood.utils import pcd_utils

        scenario_index, t_index = self._locate(dataset, idx)
        sdb = dataset.scenario_database[scenario_index]
        t_key = dataset.return_timestamp_key(sdb, t_index)
        ego_content = dataset.calc_dist_to_ego(sdb, t_key)

        data = OrderedDict()
        ego_cur_pose = None
        for cav_id, cav_content in sdb.items():
            is_ego = cav_content['ego']
            dec = self.schedule.decide(scenario_index, t_index, cav_id, is_ego)

            if not is_ego:
                self.stats['messages'] += 1
                if self.blockage is not None and self.blockage.is_blocked(
                        scenario_index, t_index, cav_id,
                        self.cfg.blockage_clearance,
                        self.cfg.blockage_min_blockers):
                    self.stats['blocked'] += 1

            if dec.drop:
                self.stats['dropped_channel'] += 1
                continue  # the message never arrived

            delay = min(dec.delay_frames, t_index)
            t_key_d = dataset.return_timestamp_key(sdb, t_index - delay)

            entry = OrderedDict()
            entry['ego'] = is_ego
            entry['time_delay'] = delay
            # reform_param mixes delayed sensing pose with CURRENT-timestamp GT
            # annotations — stock OpenCOOD semantics, exactly what we want.
            entry['params'] = dataset.reform_param(cav_content, ego_content,
                                                   t_key, t_key_d,
                                                   cur_ego_pose_flag)
            if dec.swap_seed is not None:
                entry['lidar_np'] = self._swapped_lidar(dataset, scenario_index,
                                                        dec.swap_seed)
            else:
                entry['lidar_np'] = pcd_utils.pcd_to_np(
                    cav_content[t_key_d]['lidar'])

            if dec.ghost_seed is not None:
                ghosts = self._ghost_points(dec.ghost_seed, dec.ghost_count)
                entry['lidar_np'] = np.vstack(
                    [entry['lidar_np'], ghosts]).astype(np.float32)

            if is_ego:
                ego_cur_pose = entry['params']['lidar_pose']
            else:
                if dec.pose_noise is not None:
                    self._apply_pose_noise(entry, ego_cur_pose, dec.pose_noise)
                if not self._survives_crop(dataset, entry):
                    # The impaired message would contribute ZERO points inside
                    # ego's detection crop. Stock OpenCOOD cannot represent an
                    # empty agent (the scatter builds one fewer canvas than
                    # record_len claims and warp-based fusers crash), and an
                    # empty message is indistinguishable from an absent one at
                    # fusion anyway — so drop the collaborator for this frame.
                    self.stats['dropped_empty'] += 1
                    continue

            data[cav_id] = entry
        return data

    @staticmethod
    def _survives_crop(dataset, entry, min_points=2):
        """True if the collaborator's (possibly impaired) cloud keeps at least
        min_points inside the ego-frame detection range after projection.
        Only relevant when the dataset projects points to ego first."""
        if not getattr(dataset, 'proj_first', False):
            return True
        try:
            lr = dataset.params['preprocess']['cav_lidar_range']
        except (AttributeError, KeyError, TypeError):
            return True
        tm = np.asarray(entry['params']['transformation_matrix'],
                        dtype=np.float64)
        pts = entry['lidar_np'][:, :3].astype(np.float64)
        proj = pts @ tm[:3, :3].T + tm[:3, 3]
        inside = ((proj[:, 0] > lr[0]) & (proj[:, 0] < lr[3])
                  & (proj[:, 1] > lr[1]) & (proj[:, 1] < lr[4])
                  & (proj[:, 2] > lr[2]) & (proj[:, 2] < lr[5]))
        return int(inside.sum()) >= min_points

    # -------------------------------------------------------------- impairments
    def _apply_pose_noise(self, entry, ego_cur_pose, noise):
        """Recompute the cav->ego projection from a perturbed collaborator pose.
        gt_transformation_matrix is untouched: GT stays anchored to the true pose.

        The noised position is clamped to the true pose's side of OpenCOOD's
        COM_RANGE boundary: the dataset filters CAVs by this distance but builds
        pairwise matrices from the unfiltered dict, so a membership flip crashes
        V2VNet/CoAlign (latent stock bug). Physically, meter-scale localization
        error should not change connectivity — link loss is modeled by drop."""
        from opencood.utils.transformation_utils import x1_to_x2
        assert ego_cur_pose is not None, \
            'ego must precede collaborators in scenario_database'
        pose = list(entry['params']['lidar_pose'])
        noised = [pose[0] + noise[0], pose[1] + noise[1], pose[2] + noise[2],
                  pose[3], pose[4] + noise[3], pose[5]]

        try:
            import opencood.data_utils.datasets as _od_datasets
            com_range = float(getattr(_od_datasets, 'COM_RANGE', 70.0))
        except Exception:  # noqa: BLE001 - fall back to the OPV2V default
            com_range = 70.0
        ex, ey = ego_cur_pose[0], ego_cur_pose[1]
        d_true = math.hypot(pose[0] - ex, pose[1] - ey)
        d_noised = math.hypot(noised[0] - ex, noised[1] - ey)
        margin = 0.1
        target = None
        if d_true <= com_range < d_noised:
            target = com_range - margin
        elif d_noised <= com_range < d_true:
            target = com_range + margin
        if target is not None and d_noised > 0:
            scale = target / d_noised
            noised[0] = ex + (noised[0] - ex) * scale
            noised[1] = ey + (noised[1] - ey) * scale

        entry['params']['lidar_pose'] = noised
        entry['params']['transformation_matrix'] = \
            x1_to_x2(noised, ego_cur_pose)

    def _swapped_lidar(self, dataset, scenario_index, seed):
        """Point cloud from a different scenario: the message claims this
        collaborator's pose but carries another world's content."""
        from opencood.utils import pcd_utils
        r = np.random.RandomState(seed)
        n_scen = len(dataset.scenario_database)
        choices = [i for i in range(n_scen) if i != scenario_index] or \
            [scenario_index]
        other = dataset.scenario_database[choices[r.randint(len(choices))]]
        cav = list(other.values())[r.randint(len(other))]
        t_keys = [k for k in cav.keys() if k not in ('ego', 'distance_to_ego')]
        return pcd_utils.pcd_to_np(cav[t_keys[r.randint(len(t_keys))]]['lidar'])

    def _ghost_points(self, seed, count):
        """Car-shaped point clusters in the collaborator's lidar frame: points
        sampled on the visible surfaces of a vehicle-sized box, ground-supported."""
        r = np.random.RandomState(seed)
        l, w, h = GHOST_LWH
        clusters = []
        for _ in range(count):
            radius = r.uniform(*GHOST_RANGE_XY)
            theta = r.uniform(0, 2 * np.pi)
            cx, cy = radius * np.cos(theta), radius * np.sin(theta)
            yaw = r.uniform(0, 2 * np.pi)
            n = GHOST_POINTS
            # sample on the box surface: 2 sides + roof + front/back, weighted
            faces = r.choice(5, size=n, p=[0.3, 0.3, 0.2, 0.1, 0.1])
            x = np.empty(n)
            y = np.empty(n)
            z = r.uniform(0.0, h, n)
            for f in range(5):
                m = faces == f
                k = int(m.sum())
                if f == 0:      # left side
                    x[m] = r.uniform(-l / 2, l / 2, k); y[m] = -w / 2
                elif f == 1:    # right side
                    x[m] = r.uniform(-l / 2, l / 2, k); y[m] = w / 2
                elif f == 2:    # roof
                    x[m] = r.uniform(-l / 2, l / 2, k)
                    y[m] = r.uniform(-w / 2, w / 2, k); z[m] = h
                elif f == 3:    # front
                    x[m] = l / 2; y[m] = r.uniform(-w / 2, w / 2, k)
                else:           # back
                    x[m] = -l / 2; y[m] = r.uniform(-w / 2, w / 2, k)
            x = x + r.normal(0, 0.02, n)
            y = y + r.normal(0, 0.02, n)
            c, s = np.cos(yaw), np.sin(yaw)
            xr = c * x - s * y + cx
            yr = s * x + c * y + cy
            # ground at roughly lidar_height below sensor (OPV2V lidar ~1.9m up)
            zr = z - 1.9
            intensity = r.uniform(0.2, 0.9, n)
            clusters.append(np.stack([xr, yr, zr, intensity], axis=1))
        return np.vstack(clusters).astype(np.float32)
