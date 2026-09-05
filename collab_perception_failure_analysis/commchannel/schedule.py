"""Deterministic impairment schedule.

Every random decision is derived from crc32-hashed (seed, scenario, timestamp, cav)
keys, so outcomes are reproducible and independent of DataLoader worker count or the
order in which frames are queried. The Gilbert-Elliott chain is re-simulated from the
scenario start on every query (a few hundred cheap draws), which makes burst structure
correct over time while staying stateless.
"""
import zlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def _rng(*key_parts):
    key = '/'.join(str(k) for k in key_parts)
    return np.random.RandomState(zlib.crc32(key.encode('utf8')) & 0xFFFFFFFF)


@dataclass
class Decision:
    """Per-(frame, collaborator) channel outcome. Ego always gets the default
    (perfect) decision."""
    drop: bool = False
    delay_frames: int = 0
    pose_noise: Optional[np.ndarray] = None   # [dx, dy, dz, dyaw_deg]
    ghost_seed: Optional[int] = None          # set => inject ghosts with this seed
    ghost_count: int = 0
    swap_seed: Optional[int] = None           # set => scene-swap with this seed


class Schedule:
    def __init__(self, config, blockage=None):
        self.cfg = config
        # Optional BlockageTable. Unlike every other family here, blockage is a
        # property of the SCENE, not of a random draw: the table is pure geometry
        # and the only stochastic part is P(drop | blocked).
        self.blockage = blockage

    def decide(self, scenario_index, timestamp_index, cav_id, is_ego):
        cfg = self.cfg
        d = Decision()
        if is_ego:
            return d

        # --- delivery: packet loss (i.i.d.) ---
        if cfg.loss_p > 0.0:
            r = _rng(cfg.seed, 'loss', scenario_index, timestamp_index, cav_id)
            if r.random_sample() < cfg.loss_p:
                d.drop = True
                return d

        # --- delivery: packet loss (Gilbert-Elliott bursty) ---
        if cfg.uses_gilbert_elliott:
            if self._ge_lost(scenario_index, timestamp_index, cav_id):
                d.drop = True
                return d

        # --- delivery: geometry-conditioned blockage ---
        # Same drop mechanism as i.i.d. loss, but WHICH links drop is decided by
        # scene geometry: a labeled vehicle standing on the ego<->collaborator
        # chord. Realized loss rate = blockage_p x geometric base rate, so a
        # matched-PDR i.i.d. control must be calibrated against the measured rate
        # (CommChannel.stats_dict reports it per run).
        if cfg.uses_blockage and self.blockage is not None:
            if self.blockage.is_blocked(scenario_index, timestamp_index, cav_id,
                                        cfg.blockage_clearance,
                                        cfg.blockage_min_blockers):
                if cfg.blockage_p >= 1.0:
                    d.drop = True
                    return d
                r = _rng(cfg.seed, 'blockage', scenario_index, timestamp_index,
                         cav_id)
                if r.random_sample() < cfg.blockage_p:
                    d.drop = True
                    return d

        # --- delivery: constant latency / content: stale memory ---
        # Latency: message is k frames old. Stale: message refreshes every N frames,
        # so age grows 0..N-1 then snaps back (sawtooth). Both compose additively.
        delay = cfg.latency_frames
        if cfg.stale_period > 1:
            delay += timestamp_index % cfg.stale_period
        d.delay_frames = delay

        # --- content: pose noise ---
        if cfg.pose_xyz_std > 0.0 or cfg.pose_yaw_std_deg > 0.0:
            r = _rng(cfg.seed, 'pose', scenario_index, timestamp_index, cav_id)
            noise = np.zeros(4)
            noise[:3] = r.normal(0.0, cfg.pose_xyz_std, 3) \
                if cfg.pose_xyz_std > 0.0 else 0.0
            noise[3] = r.normal(0.0, cfg.pose_yaw_std_deg) \
                if cfg.pose_yaw_std_deg > 0.0 else 0.0
            d.pose_noise = noise

        # --- content: ghost vehicles ---
        if cfg.ghost_p > 0.0:
            r = _rng(cfg.seed, 'ghost', scenario_index, timestamp_index, cav_id)
            if r.random_sample() < cfg.ghost_p:
                d.ghost_seed = int(r.randint(0, 2 ** 31 - 1))
                d.ghost_count = cfg.ghost_count

        # --- content: scene swap ---
        if cfg.swap_p > 0.0:
            r = _rng(cfg.seed, 'swap', scenario_index, timestamp_index, cav_id)
            if r.random_sample() < cfg.swap_p:
                d.swap_seed = int(r.randint(0, 2 ** 31 - 1))

        return d

    def _ge_lost(self, scenario_index, timestamp_index, cav_id):
        """Simulate the 2-state chain from the scenario start; returns loss at t."""
        cfg = self.cfg
        r = _rng(cfg.seed, 'ge', scenario_index, cav_id)
        state_bad = False
        lost = False
        for _ in range(timestamp_index + 1):
            # transition first, then draw loss in the current state
            if state_bad:
                if r.random_sample() < cfg.ge_p_bad_to_good:
                    state_bad = False
            else:
                if r.random_sample() < cfg.ge_p_good_to_bad:
                    state_bad = True
            p_loss = cfg.ge_loss_bad if state_bad else cfg.ge_loss_good
            lost = r.random_sample() < p_loss
        return lost
