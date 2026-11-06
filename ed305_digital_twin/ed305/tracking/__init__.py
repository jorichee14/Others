"""Sensor-agnostic multi-person tracking in the room frame (see TRACKING.md).

Every sensor gives Measurements (time, floor position + covariance, optional
appearance feature or known identity) and has a SensorModel (where and how
reliably it detects people, how often it reports clutter). The tracker
builds short, safe tracklets online and joins them into identities with one
global assignment (motion, appearance, identity, entry/exit zones).
"""
from .model import Measurement, SensorModel, Zones, load_config
from .tracker import TrackerConfig, Track, Tracklet, build_tracklets, link_tracklets, track
from .metrics import id_metrics

__all__ = ["Measurement", "SensorModel", "Zones", "load_config", "TrackerConfig", "Track", "Tracklet",
           "build_tracklets", "link_tracklets", "track", "id_metrics"]
