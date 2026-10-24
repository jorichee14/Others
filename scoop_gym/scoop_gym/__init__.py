"""SCooP-iT Gym: multi-agent environments on the measured link of the SCooP-iT
testbed (README.md). CommSchedule-v0 is registered with gymnasium, when it is
installed, as a single-sender view (scoop_gym.comm_schedule.gymnasium_env)."""
from .comm_schedule import CommSchedule, gymnasium_env      # noqa: F401
from .link import LinkEngine, Message                       # noqa: F401
from .record import Recorder                                # noqa: F401

try:
    import gymnasium as _gym

    def _single(**kw):
        return gymnasium_env(CommSchedule(**kw))

    if "scoop/CommSchedule-v0" not in _gym.registry:
        _gym.register("scoop/CommSchedule-v0", entry_point=_single)
except ImportError:                                         # gymnasium is optional
    pass
