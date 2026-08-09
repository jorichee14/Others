from .config import ChannelConfig
from .schedule import Schedule, Decision
from .channel import CommChannel
from .blockage import BlockageTable, locate_frame

__all__ = ['ChannelConfig', 'Schedule', 'Decision', 'CommChannel',
           'BlockageTable', 'locate_frame', 'attach_bandwidth_hooks']


def attach_bandwidth_hooks(*args, **kwargs):
    """Lazy proxy: feature_hooks needs torch, which the data-level channel doesn't."""
    from .feature_hooks import attach_bandwidth_hooks as _impl
    return _impl(*args, **kwargs)
