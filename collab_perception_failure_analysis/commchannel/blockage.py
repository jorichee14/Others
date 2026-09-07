"""Geometry-conditioned link blockage: which collaborator messages are obstructed
by the labeled vehicles standing between the two agents.

Every impairment family in this study so far drops messages *independently* of the
scene. That is the field's standard assumption and it is physically wrong: the
vehicles that occlude an agent's lidar are the same vehicles that obstruct its
radio. This module measures the geometry so the assumption can be tested rather
than inherited.

A collaborator's link at (scenario, timestamp) is BLOCKED at clearance `c` when at
least one labeled vehicle box, inflated by `c` metres, intersects the 2-D segment
between the ego's and the collaborator's `lidar_pose`. Clearance stands in for the
first Fresnel radius (~1.1 m for 5.9 GHz over 50 m), so it is swept rather than
fixed — `BlockageTable` stores blocker counts for a whole grid of clearances in one
pass.

The table is built from OPV2V yaml alone (poses + `vehicles`); no point clouds are
read, so a full test-split build is a few minutes and caches to disk.

NOTE ON SCOPE: this module answers "is the chord geometrically obstructed", not
"what is the path loss". No propagation model, no dB, no packet-delivery mapping —
those belong downstream, and mixing them in here would let a modelling assumption
manufacture the very correlation the audit is trying to measure.
"""
import json
import math
import os
import zlib

import numpy as np

DEFAULT_CLEARANCES = (0.0, 1.0, 2.0)
# A blocker box whose centre is nearer than this (m) to either chord endpoint is
# taken to BE that endpoint's vehicle. OPV2V lists the CAVs themselves in the
# `vehicles` dict, and a CAV always sits on its own lidar_pose.
DEFAULT_ENDPOINT_MARGIN = 2.0


# --------------------------------------------------------------------- geometry
def _segment_aabb_hit(q0, q1, half):
    """Liang-Barsky slab clip: does segment q0->q1 meet the origin-centred
    axis-aligned box with half-extents `half`? All 2-vectors."""
    t0, t1 = 0.0, 1.0
    d = q1 - q0
    for k in (0, 1):
        if abs(d[k]) < 1e-12:
            if abs(q0[k]) > half[k]:
                return False
            continue
        inv = 1.0 / d[k]
        ta = (-half[k] - q0[k]) * inv
        tb = (half[k] - q0[k]) * inv
        if ta > tb:
            ta, tb = tb, ta
        t0 = max(t0, ta)
        t1 = min(t1, tb)
        if t0 > t1:
            return False
    return True


def _to_box_frame(p, center, yaw_deg):
    a = math.radians(yaw_deg)
    c, s = math.cos(a), math.sin(a)
    d = np.asarray(p, dtype=np.float64)[:2] - np.asarray(center, dtype=np.float64)[:2]
    return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])


def segment_blocked_by_box(p0, p1, center, yaw_deg, half_xy, clearance=0.0):
    """Does the chord p0->p1 intersect an oriented box inflated by `clearance`?"""
    q0 = _to_box_frame(p0, center, yaw_deg)
    q1 = _to_box_frame(p1, center, yaw_deg)
    half = np.asarray(half_xy, dtype=np.float64) + float(clearance)
    return _segment_aabb_hit(q0, q1, half)


def point_in_box(p, center, yaw_deg, half_xy, margin=0.0):
    q = np.abs(_to_box_frame(p, center, yaw_deg))
    half = np.asarray(half_xy, dtype=np.float64) + float(margin)
    return bool(q[0] <= half[0] and q[1] <= half[1])


def boxes_from_params(params):
    """[(vehicle_id, centre_xy, yaw_deg, half_xy)] from an OPV2V yaml param dict.

    OPV2V/CARLA convention: `extent` is a HALF-extent, `angle` is
    [roll, yaw, pitch] in degrees, and the box centre is `location` offset by
    `center` (small, and mostly vertical, but applied for correctness)."""
    out = []
    for vid, v in (params.get('vehicles') or {}).items():
        try:
            loc = np.asarray(v['location'], dtype=np.float64)
            ext = np.asarray(v['extent'], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            continue
        ctr = np.asarray(v.get('center', [0.0, 0.0, 0.0]), dtype=np.float64)
        ang = v.get('angle', [0.0, 0.0, 0.0])
        yaw = float(ang[1]) if len(ang) > 1 else 0.0
        out.append((vid, loc[:2] + ctr[:2], yaw, ext[:2]))
    return out


def count_blockers(p0, p1, boxes, clearances, endpoint_margin=DEFAULT_ENDPOINT_MARGIN):
    """Blockers on the chord p0->p1 at each clearance in `clearances`.

    Boxes containing (or centred within `endpoint_margin` of) either endpoint are
    the endpoint vehicles themselves and are excluded — otherwise every link would
    report itself blocked by its own two cars."""
    counts = [0] * len(clearances)
    for _vid, center, yaw, half in boxes:
        if (point_in_box(p0, center, yaw, half)
                or point_in_box(p1, center, yaw, half)):
            continue
        if (np.linalg.norm(np.asarray(center) - np.asarray(p0)[:2]) < endpoint_margin
                or np.linalg.norm(np.asarray(center) - np.asarray(p1)[:2])
                < endpoint_margin):
            continue
        for ci, c in enumerate(clearances):
            if segment_blocked_by_box(p0, p1, center, yaw, half, c):
                counts[ci] += 1
    return counts


# ------------------------------------------------------------------ frame index
def locate_frame(dataset, idx):
    """Global dataset index -> (scenario_index, timestamp_index). Mirrors stock
    OpenCOOD's own lookup, and is the key space `Schedule.decide` works in."""
    scenario_index = 0
    for i, ele in enumerate(dataset.len_record):
        if idx < ele:
            scenario_index = i
            break
    t_index = idx if scenario_index == 0 else \
        idx - dataset.len_record[scenario_index - 1]
    return scenario_index, t_index


# --------------------------------------------------------------------- the table
class BlockageTable:
    """(scenario_index, timestamp_index, cav_id) -> blocker count per clearance.

    Deterministic given the dataset: pure geometry, no random draws."""

    def __init__(self, counts, clearances, meta=None):
        self.counts = counts
        self.clearances = tuple(float(c) for c in clearances)
        self.meta = meta or {}

    # ------------------------------------------------------------ query
    def _index_for(self, clearance):
        for i, c in enumerate(self.clearances):
            if abs(c - float(clearance)) < 1e-9:
                return i
        raise KeyError(
            'clearance %r not in table grid %r — rebuild the table with this '
            'clearance included' % (clearance, self.clearances))

    def n_blockers(self, scenario_index, timestamp_index, cav_id, clearance):
        row = self.counts.get((scenario_index, timestamp_index, str(cav_id)))
        if row is None:
            return 0
        return int(row[self._index_for(clearance)])

    def is_blocked(self, scenario_index, timestamp_index, cav_id, clearance,
                   min_blockers=1):
        return self.n_blockers(scenario_index, timestamp_index, cav_id,
                               clearance) >= int(min_blockers)

    def base_rate(self, clearance, min_blockers=1):
        """Fraction of (frame, collaborator) links that are blocked."""
        if not self.counts:
            return 0.0
        ci = self._index_for(clearance)
        hit = sum(1 for row in self.counts.values() if row[ci] >= min_blockers)
        return hit / len(self.counts)

    # ------------------------------------------------------------ build
    @classmethod
    def build(cls, dataset, clearances=DEFAULT_CLEARANCES,
              endpoint_margin=DEFAULT_ENDPOINT_MARGIN, verbose=True):
        """Walk every (scenario, timestamp, collaborator) reading yaml only.

        The blocker set is the UNION of the ego's and the collaborator's own
        `vehicles` lists: either agent may be the only one that labelled a car
        sitting midway along the chord."""
        from opencood.hypes_yaml import yaml_utils

        clearances = tuple(float(c) for c in clearances)
        counts = {}
        n_scen = len(dataset.scenario_database)
        for si in range(n_scen):
            sdb = dataset.scenario_database[si]
            ego_id = next((cid for cid, c in sdb.items() if c.get('ego')), None)
            if ego_id is None:
                continue
            n_t = len(dataset.len_record) and (
                dataset.len_record[si] - (dataset.len_record[si - 1] if si else 0))
            for ti in range(n_t):
                t_key = dataset.return_timestamp_key(sdb, ti)
                try:
                    ego_params = yaml_utils.load_yaml(sdb[ego_id][t_key]['yaml'])
                except (KeyError, OSError):
                    continue
                p_ego = np.asarray(ego_params['lidar_pose'], dtype=np.float64)[:2]
                ego_boxes = boxes_from_params(ego_params)

                for cav_id, cav_content in sdb.items():
                    if cav_id == ego_id:
                        continue
                    try:
                        cav_params = yaml_utils.load_yaml(
                            cav_content[t_key]['yaml'])
                    except (KeyError, OSError):
                        continue
                    p_cav = np.asarray(cav_params['lidar_pose'],
                                       dtype=np.float64)[:2]
                    merged = dict((b[0], b) for b in ego_boxes)
                    merged.update((b[0], b) for b in boxes_from_params(cav_params))
                    counts[(si, ti, str(cav_id))] = count_blockers(
                        p_ego, p_cav, list(merged.values()), clearances,
                        endpoint_margin)
            if verbose:
                print('[blockage] scenario %d/%d — %d links'
                      % (si + 1, n_scen, len(counts)))
        meta = {
            'clearances': list(clearances),
            'endpoint_margin': endpoint_margin,
            'n_links': len(counts),
            'n_scenarios': n_scen,
        }
        return cls(counts, clearances, meta)

    # ------------------------------------------------------------ persistence
    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            'clearances': list(self.clearances),
            'meta': self.meta,
            'counts': {'%d|%d|%s' % k: v for k, v in self.counts.items()},
        }
        with open(path, 'w') as f:
            json.dump(payload, f)
        return path

    @classmethod
    def load(cls, path):
        with open(path) as f:
            payload = json.load(f)
        counts = {}
        for k, v in payload['counts'].items():
            si, ti, cav = k.split('|', 2)
            counts[(int(si), int(ti), cav)] = v
        return cls(counts, payload['clearances'], payload.get('meta'))

    @classmethod
    def load_or_build(cls, dataset, clearances=DEFAULT_CLEARANCES,
                      endpoint_margin=DEFAULT_ENDPOINT_MARGIN,
                      cache_dir=None, verbose=True):
        """Disk-cached build. The key covers dataset identity and build params, so
        a different split or clearance grid never silently reuses a stale table."""
        cache_dir = cache_dir or os.path.expanduser('~/.cache/cpfa_blockage')
        key = zlib.crc32(('|'.join([
            str(getattr(dataset, 'root_dir', '?')),
            str(len(dataset.scenario_database)),
            str(list(getattr(dataset, 'len_record', []))),
            str(list(clearances)), str(endpoint_margin),
        ])).encode('utf8')) & 0xFFFFFFFF
        path = os.path.join(cache_dir, 'blockage_%08x.json' % key)
        if os.path.isfile(path):
            if verbose:
                print('[blockage] cache hit %s' % path)
            return cls.load(path)
        table = cls.build(dataset, clearances, endpoint_margin, verbose)
        table.save(path)
        if verbose:
            print('[blockage] cached -> %s' % path)
        return table


# ------------------------------------------------------------------- self-test
def _selftest():
    """Geometry checks that need no dataset, no opencood, and no GPU."""
    ok = True

    def check(name, got, want):
        nonlocal ok
        if got != want:
            ok = False
            print('  FAIL %-46s got %r want %r' % (name, got, want))
        else:
            print('  ok   %-46s %r' % (name, got))

    # a 4.5 x 1.9 m car (half-extents 2.25 x 0.95) sitting at the origin
    half = (2.25, 0.95)
    check('car astride the chord blocks it',
          segment_blocked_by_box((-10, 0), (10, 0), (0, 0), 0.0, half), True)
    check('car 5 m off the chord does not',
          segment_blocked_by_box((-10, 5), (10, 5), (0, 0), 0.0, half), False)
    check('car 1.5 m off: clear at c=0',
          segment_blocked_by_box((-10, 1.5), (10, 1.5), (0, 0), 0.0, half), False)
    check('car 1.5 m off: blocked at c=1',
          segment_blocked_by_box((-10, 1.5), (10, 1.5), (0, 0), 0.0, half, 1.0),
          True)
    check('rotating the car 90 deg re-blocks a near-miss chord',
          segment_blocked_by_box((-10, 1.5), (10, 1.5), (0, 0), 90.0, half), True)
    check('chord stopping short of the car misses',
          segment_blocked_by_box((-10, 0), (-5, 0), (0, 0), 0.0, half), False)
    check('point inside box', point_in_box((0.5, 0.2), (0, 0), 0.0, half), True)
    check('point outside box', point_in_box((9.0, 0.2), (0, 0), 0.0, half), False)

    # endpoint exclusion: two cars ON the endpoints plus one real blocker midway
    boxes = [
        ('ego_car', np.array([0.0, 0.0]), 0.0, half),
        ('cav_car', np.array([40.0, 0.0]), 0.0, half),
        ('blocker', np.array([20.0, 0.0]), 0.0, half),
    ]
    check('endpoint vehicles excluded, real blocker counted',
          count_blockers((0.0, 0.0), (40.0, 0.0), boxes, (0.0, 1.0)), [1, 1])
    boxes_far = [('blocker', np.array([20.0, 3.0]), 0.0, half)]
    check('blocker 3 m off chord: clear at 0/1, blocked at 2.2',
          count_blockers((0.0, 0.0), (40.0, 0.0), boxes_far, (0.0, 1.0, 2.2)),
          [0, 0, 1])

    # yaml parsing + table round-trip
    params = {'vehicles': {
        7: {'location': [20.0, 0.0, 0.0], 'extent': [2.25, 0.95, 0.75],
            'angle': [0.0, 0.0, 0.0], 'center': [0.0, 0.0, 0.5]},
    }}
    parsed = boxes_from_params(params)
    check('boxes_from_params yields one box', len(parsed), 1)
    check('half-extent read as half', tuple(parsed[0][3]), (2.25, 0.95))

    t = BlockageTable({(0, 3, '641'): [0, 1, 2]}, (0.0, 1.0, 2.0))
    check('is_blocked at c=0', t.is_blocked(0, 3, '641', 0.0), False)
    check('is_blocked at c=1', t.is_blocked(0, 3, '641', 1.0), True)
    check('min_blockers=2 at c=1', t.is_blocked(0, 3, '641', 1.0, 2), False)
    check('unknown link reads as clear', t.is_blocked(9, 9, 'zz', 1.0), False)
    check('base_rate at c=2', round(t.base_rate(2.0), 3), 1.0)
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = t.save(os.path.join(d, 'tbl.json'))
        t2 = BlockageTable.load(p)
        check('round-trip preserves counts', t2.counts, t.counts)
        check('round-trip preserves grid', t2.clearances, t.clearances)

    print('\nselftest: %s' % ('PASS' if ok else 'FAIL'))
    return 0 if ok else 1


if __name__ == '__main__':
    import sys
    sys.exit(_selftest())
