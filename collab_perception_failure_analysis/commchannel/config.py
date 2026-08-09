"""Channel impairment configuration.

A ChannelConfig holds the parameters for every impairment family; each is inactive
unless its parameters are set. Impairments compose (e.g. latency + packet loss).

Delivery impairments (message never usefully arrives):
    latency_frames   : constant delay of collaborator messages, in frames (10Hz data).
    loss_p           : i.i.d. Bernoulli packet loss probability per (frame, collaborator).
    ge_*             : Gilbert-Elliott bursty loss (2-state Markov chain per collaborator
                       per scenario; drops occur in the Bad state).
    blockage_p       : P(drop | the ego<->collaborator chord is geometrically blocked
                       by a labeled vehicle). Loss is SCENE-CONDITIONED rather than
                       i.i.d.: the realized loss rate is blockage_p x the geometric
                       base rate, so it must be measured, not assumed (see
                       docs/BLOCKAGE.md and scripts/run_blockage_audit.py).
    blockage_clearance      : metres the blocker box is inflated by before the chord
                       test — a stand-in for the first Fresnel radius.
    blockage_min_blockers   : how many vehicles must sit on the chord to call it
                       blocked (1 = any).
    bandwidth_bits   : uniform quantization bit-width of shared features (32 = off).
                       Applied via model hooks, not the dataset wrapper.

Content impairments (message arrives, contents are wrong):
    stale_period     : collaborator messages refresh only every N frames; between
                       refreshes the last transmitted frame is re-delivered.
    pose_xyz_std     : Gaussian noise (meters) on collaborator position in its
                       reported pose (spatial misalignment at fusion).
    pose_yaw_std_deg : Gaussian noise (degrees) on collaborator yaw.
    ghost_p          : probability per (frame, collaborator) of injecting ghost
                       vehicles (car-shaped point clusters) into its point cloud.
    ghost_count      : number of ghost vehicles per event.
    swap_p           : probability per (frame, collaborator) of replacing its point
                       cloud with one from a DIFFERENT scenario (pure content conflict).

seed drives every random decision; identical config + seed => identical impairment
realization, independent of DataLoader worker count or query order.
"""
from dataclasses import dataclass, asdict, field


@dataclass
class ChannelConfig:
    # delivery
    latency_frames: int = 0
    loss_p: float = 0.0
    ge_p_good_to_bad: float = 0.0
    ge_p_bad_to_good: float = 0.0
    ge_loss_good: float = 0.0
    ge_loss_bad: float = 1.0
    blockage_p: float = 0.0
    blockage_clearance: float = 1.0
    blockage_min_blockers: int = 1
    bandwidth_bits: int = 32
    # content
    stale_period: int = 0
    pose_xyz_std: float = 0.0
    pose_yaw_std_deg: float = 0.0
    ghost_p: float = 0.0
    ghost_count: int = 3
    swap_p: float = 0.0
    # bookkeeping
    seed: int = 0
    name: str = 'identity'

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    @property
    def is_identity(self):
        return (self.latency_frames == 0 and self.loss_p == 0.0
                and self.ge_p_good_to_bad == 0.0 and self.bandwidth_bits >= 32
                and self.stale_period == 0 and self.pose_xyz_std == 0.0
                and self.pose_yaw_std_deg == 0.0 and self.ghost_p == 0.0
                and self.swap_p == 0.0 and self.blockage_p == 0.0)

    @property
    def uses_gilbert_elliott(self):
        return self.ge_p_good_to_bad > 0.0

    @property
    def uses_blockage(self):
        return self.blockage_p > 0.0
