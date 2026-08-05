from .config import ChannelConfig
from .schedule import Schedule, Decision
from .channel import CommChannel

__all__ = ['ChannelConfig', 'Schedule', 'Decision', 'CommChannel',
           'attach_bandwidth_hooks']


def attach_bandwidth_hooks(*args, **kwargs):
    """Lazy proxy: feature_hooks needs torch, which the data-level channel doesn't."""
    from .feature_hooks import attach_bandwidth_hooks as _impl
    return _impl(*args, **kwargs)
