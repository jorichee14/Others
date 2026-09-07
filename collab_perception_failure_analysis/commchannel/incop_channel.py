"""IncopChannel: impairment wrapper for the InCoP / HEAL data model.

WHY A SEPARATE CLASS. `channel.py` reimplements OpenCOOD's `retrieve_base_data` so it
can substitute a past timestamp, and that reimplementation calls `dataset.reform_param`
and `dataset.calc_dist_to_ego`. **Neither method exists in the HEAL lineage** that
InCoP is built on — they are DerrickXuNu-OpenCOOD constructs. `CommChannel` therefore
cannot be attached to an InCoP dataset, despite the hook point (`retrieve_base_data`)
being identical.

HOW THIS ONE WORKS INSTEAD. Rather than reimplement HEAL's loader, it *calls* it twice
and splices:

    past = orig_retrieve(idx - delay)   # sensing: lidar, cameras, reported pose
    cur  = orig_retrieve(idx)           # ground truth: object annotations

That single decision buys correctness for free. HEAL's loader handles json-vs-yaml
params, hdf5-vs-png cameras, depth, `add_data_extension`, and — for InCoP specifically —
`IsaacSimBaseDataset`'s camera-extrinsic normalisation and image resizing, which a
reimplementation would have silently skipped, producing subtly wrong camera geometry
under every impairment. The cost is one extra load per distinct delay value in a frame
(normally one, since the study applies impairments uniformly), hidden by DataLoader
workers.

GT POLICY. Identical in spirit to `channel.py`: annotations always come from the CURRENT
timestamp regardless of delay. In HEAL the object annotations live under
`params['vehicles']`, and `IsaacSimBaseDataset.generate_object_center_*` unions that key
across CAVs — so a delayed collaborator would otherwise inject stale GT into the union.

POSE NOISE is simpler here than in OpenCOOD, and better. HEAL's fusion dataset computes
`transformation_matrix = x1_to_x2(lidar_pose, ego_pose)` and a separate
`transformation_matrix_clean` from `lidar_pose_clean`. So perturbing `lidar_pose` alone
produces the misalignment while GT stays anchored to `lidar_pose_clean` automatically.
No COM_RANGE clamp is needed either: InCoP's `comm_range` (50 m) far exceeds its ~22 m
scene extent, so no perturbation can flip connectivity.

GHOSTS are rescaled to the indoor benchmark. A 4.5 x 1.9 x 1.6 m sedan dropped into a
hospital corridor is not a plausible false positive; InCoP's anchors are 1 x 1 x 1 m and
its classes are chairs, trash cans, and fire extinguishers.

Bandwidth quantization is unchanged — `feature_hooks.py` already registers
`HeterModelBevfusionHighresIsaac`.
"""
from collections import OrderedDict

import numpy as np

from .blockage import locate_frame
from .channel import CommChannel

# Object annotation keys taken from the CURRENT frame regardless of message delay.
GT_KEYS = ('vehicles',)

# Indoor ghost geometry: a chair/trash-can-sized box, not a sedan.
INDOOR_GHOST_LWH = (0.6, 0.6, 1.0)
INDOOR_GHOST_POINTS = 60
INDOOR_GHOST_RANGE_XY = (1.5, 10.0)


class IncopChannel(CommChannel):
    """Impairments for InCoP/HEAL datasets (IsaacSim indoor, or HEAL's OPV2V)."""

    def __init__(self, config, blockage=None, indoor_ghosts=True):
        super().__init__(config, blockage=blockage)
        self.indoor_ghosts = indoor_ghosts

    # ------------------------------------------------------------------ attach
    def attach(self, dataset):
        if self._attached:
            raise RuntimeError('channel already attached')
        if self.cfg.uses_blockage:
            raise NotImplementedError(
                'blockage is not ported to InCoP: BlockageTable assumes OPV2V vehicle '
                'annotations and outdoor chord geometry. It was also a NEGATIVE result '
                'on OPV2V (see docs/BLOCKAGE.md) and should not be revived here without '
                'a reason.')
        self._dataset = dataset
        self._orig_retrieve = dataset.retrieve_base_data
        channel = self

        def impaired_retrieve(idx, cur_ego_pose_flag=True):
            return channel._retrieve(dataset, idx, cur_ego_pose_flag)

        dataset.retrieve_base_data = impaired_retrieve
        self._attached = True
        return dataset

    # ------------------------------------------------------------ core retrieve
    def _retrieve(self, dataset, idx, cur_ego_pose_flag=True):
        scenario_index, t_index = locate_frame(dataset, idx)
        sdb = dataset.scenario_database[scenario_index]

        cur = self._orig_retrieve(idx)
        past_cache = {0: cur}

        data = OrderedDict()
        ego_cur_pose = None

        for cav_id, cav_content in sdb.items():
            is_ego = bool(cav_content['ego'])
            dec = self.schedule.decide(scenario_index, t_index, cav_id, is_ego)

            if not is_ego:
                self.stats['messages'] += 1
            if dec.drop:
                self.stats['dropped_channel'] += 1
                continue
            if cav_id not in cur:
                # The stock loader already excluded this CAV (out of comm range,
                # or absent at this timestamp). Nothing to impair.
                continue

            # Delay is clamped to the start of the scenario: within-scenario sample
            # indices are contiguous, so idx - delay stays in the same scenario.
            delay = int(min(dec.delay_frames, t_index))
            if delay not in past_cache:
                past_cache[delay] = self._orig_retrieve(idx - delay)
            past = past_cache[delay]
            source = past.get(cav_id, cur[cav_id])

            entry = OrderedDict(source)
            entry['ego'] = is_ego
            entry['time_delay'] = delay

            # ---- GT policy: annotations always from the CURRENT timestamp -------
            entry['params'] = dict(source['params'])
            cur_params = cur[cav_id]['params']
            for key in GT_KEYS:
                if key in cur_params:
                    entry['params'][key] = cur_params[key]
                else:
                    entry['params'].pop(key, None)

            if dec.swap_seed is not None:
                entry['lidar_np'] = self._swapped_lidar_incop(
                    dataset, scenario_index, dec.swap_seed)

            if dec.ghost_seed is not None:
                ghosts = self._ghosts(dec.ghost_seed, dec.ghost_count)
                entry['lidar_np'] = np.vstack(
                    [entry['lidar_np'], ghosts]).astype(np.float32)

            if is_ego:
                ego_cur_pose = entry['params']['lidar_pose']
            elif dec.pose_noise is not None:
                self._apply_pose_noise_incop(entry, dec.pose_noise)

            data[cav_id] = entry

        assert ego_cur_pose is not None or not data, \
            'ego missing from the impaired sample — the ego is never dropped'
        return data

    # -------------------------------------------------------------- impairments
    @staticmethod
    def _apply_pose_noise_incop(entry, noise):
        """Perturb the collaborator's REPORTED pose only.

        HEAL derives `transformation_matrix` from `lidar_pose` and
        `transformation_matrix_clean` from `lidar_pose_clean`, so this produces the
        misalignment while leaving the GT-anchored transform untouched. No COM_RANGE
        clamp: InCoP's comm_range (50 m) dwarfs its ~22 m scene extent, so a
        metre-scale perturbation cannot flip connectivity the way it did on OPV2V.
        """
        pose = list(entry['params']['lidar_pose'])
        entry['params']['lidar_pose'] = [
            pose[0] + noise[0], pose[1] + noise[1], pose[2] + noise[2],
            pose[3], pose[4] + noise[3], pose[5],
        ]
        entry['params'].setdefault('lidar_pose_clean', list(pose))

    def _ghosts(self, seed, count):
        if not self.indoor_ghosts:
            return self._ghost_points(seed, count)
        r = np.random.RandomState(seed)
        l, w, h = INDOOR_GHOST_LWH
        clusters = []
        for _ in range(count):
            radius = r.uniform(*INDOOR_GHOST_RANGE_XY)
            theta = r.uniform(0, 2 * np.pi)
            cx, cy = radius * np.cos(theta), radius * np.sin(theta)
            yaw = r.uniform(0, 2 * np.pi)
            n = INDOOR_GHOST_POINTS
            # Uniform on the box surface is good enough at this scale; the point of a
            # ghost is a plausible cluster where nothing is, not a rendered object.
            local = np.stack([
                r.uniform(-l / 2, l / 2, n),
                r.uniform(-w / 2, w / 2, n),
                r.uniform(0.0, h, n),
            ], axis=1)
            c, s = np.cos(yaw), np.sin(yaw)
            pts = np.empty((n, 4), dtype=np.float32)
            pts[:, 0] = cx + local[:, 0] * c - local[:, 1] * s
            pts[:, 1] = cy + local[:, 0] * s + local[:, 1] * c
            pts[:, 2] = local[:, 2]
            pts[:, 3] = r.uniform(0.0, 1.0, n)
            clusters.append(pts)
        return np.vstack(clusters).astype(np.float32)

    @staticmethod
    def _swapped_lidar_incop(dataset, scenario_index, seed):
        """Cloud from a different scenario: the message claims this collaborator's
        pose but carries another scene's content. Uses the dataset's own loader entry
        rather than assuming a .pcd path, since InCoP data may be stored differently."""
        from opencood.utils import pcd_utils
        r = np.random.RandomState(seed)
        n_scen = len(dataset.scenario_database)
        choices = [i for i in range(n_scen) if i != scenario_index] or [scenario_index]
        other = dataset.scenario_database[choices[r.randint(len(choices))]]
        cav = list(other.values())[r.randint(len(other))]
        t_keys = [k for k in cav.keys()
                  if k not in ('ego', 'distance_to_ego') and isinstance(cav[k], dict)
                  and 'lidar' in cav[k]]
        if not t_keys:
            raise RuntimeError('scene swap: no timestamped lidar entries found')
        return pcd_utils.pcd_to_np(cav[t_keys[r.randint(len(t_keys))]]['lidar'])
