# -*- coding: utf-8 -*-
"""
rosbag2 -> OPV2V conversion.

Output layout (exactly what ``opencood.data_utils.datasets.basedataset`` walks)::

    <root>/<split>/<scenario>/<cav_id>/<000000>.pcd
                                      /<000000>.yaml
                                      /<000000>_camera0.png   (optional)

The pipeline is three stages, in this order for a reason:

1. **Index** — stamps only, no deserialisation (:meth:`BagReader.index`).
2. **Plan** — frame times, per-agent message matching, pose resolution.  Every
   frame that cannot be fully resolved is dropped *here*, before anything is
   written, because OpenCOOD indexes all agents by the ego's timestamp keys and a
   half-populated frame is a ``KeyError`` at training time.
3. **Write** — one streaming pass that decodes only the chosen messages.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from . import clock as clockmod
from . import labels
from .bagreader import BagReader
from .config import AgentConfig, ConverterConfig, ego_agent
from .geometry import invert, matrix_to_opencood_pose, quat_to_matrix
from .pointclouds import (apply_range_filter, cloud_from_depth_image,
                          cloud_from_pointcloud2, camera_intrinsics, deskew_cloud,
                          subsample)
from .sync import (NS, Frame, FrameTable, PoseTrack, StampIndex,
                   build_frame_table, frame_times, reuse_statistics,
                   tightness_curve)
from .writers import image_to_array, write_frame_yaml, write_pcd, write_png


class ConversionError(RuntimeError):
    pass


@dataclass
class ConversionReport:
    """Everything worth knowing about a run, for the console and provenance file."""
    bag: str = ""
    output: str = ""
    scenarios: List[str] = field(default_factory=list)
    frames_written: int = 0
    frames_candidate: int = 0
    dropped: Dict[str, int] = field(default_factory=dict)
    dropped_runs: List[dict] = field(default_factory=list)
    points_per_agent: Dict[str, dict] = field(default_factory=dict)
    sync: Dict[str, dict] = field(default_factory=dict)
    pose_stats: Dict[str, dict] = field(default_factory=dict)
    clocks: Dict[str, dict] = field(default_factory=dict)
    tightness: Dict[str, object] = field(default_factory=dict)
    deskew: Dict[str, dict] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    duration_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "bag": self.bag,
            "output": self.output,
            "scenarios": self.scenarios,
            "frames_written": self.frames_written,
            "frames_candidate": self.frames_candidate,
            "dropped": self.dropped,
            "dropped_runs": self.dropped_runs,
            "points_per_agent": self.points_per_agent,
            "sync": self.sync,
            "pose_stats": self.pose_stats,
            "clocks": self.clocks,
            "tightness": self.tightness,
            "deskew": self.deskew,
            "warnings": self.warnings,
            "duration_s": round(self.duration_s, 2),
        }


# --------------------------------------------------------------------- helpers

def _pose_and_speed(msg) -> Tuple[np.ndarray, float]:
    """Extract ``(4x4 transform, speed m/s)`` from an odometry-like message."""
    pose = getattr(msg, "pose", None)
    if pose is None:
        raise ConversionError("pose topic message has no 'pose' field")
    inner = getattr(pose, "pose", pose)          # Odometry / PoseWithCovariance
    position = inner.position
    orientation = inner.orientation

    transform = quat_to_matrix(float(orientation.x), float(orientation.y),
                               float(orientation.z), float(orientation.w))
    transform[:3, 3] = [float(position.x), float(position.y), float(position.z)]

    speed = 0.0
    twist = getattr(msg, "twist", None)
    if twist is not None:
        linear = getattr(getattr(twist, "twist", twist), "linear", None)
        if linear is not None:
            speed = float(np.linalg.norm([linear.x, linear.y, linear.z]))
    return transform, speed


def _fill_speed_from_motion(track: PoseTrack) -> bool:
    """Derive speed by differencing positions when the bag carries no twist.

    Returns True if speeds were replaced.  ``ego_speed`` is only used by
    intermediate fusion (normalised by 30 km/h), so a wrong-but-plausible zero is
    worse than a derived value.
    """
    stamps = track.stamps
    if len(stamps) < 2 or any(abs(s) > 1e-9 for s in track.speeds):
        return False
    positions = np.array([t[:3, 3] for t in track.transforms])
    times = np.asarray(stamps, dtype=np.float64) / NS
    speeds = [0.0] * len(stamps)
    for i in range(1, len(stamps)):
        dt = times[i] - times[i - 1]
        if dt > 1e-6:
            speeds[i] = float(np.linalg.norm(positions[i] - positions[i - 1]) / dt)
    speeds[0] = speeds[1] if len(speeds) > 1 else 0.0
    track.set_speeds(speeds)
    return True


def _sensor_pose(world_from_base: np.ndarray, extrinsic: np.ndarray,
                 ground_lift: float) -> np.ndarray:
    """World pose of a sensor, including the ground-lift compensation.

    ``ground_lift`` shifts the *points* down inside the sensor frame and the
    sensor *pose* up by the same amount, so the world position of every point is
    unchanged while the apparent sensor height above the floor grows.  That is the
    knob for making a knee-high robot LiDAR look to a car-trained detector like a
    roof-mounted one.
    """
    pose = world_from_base @ extrinsic
    if ground_lift:
        pose = pose.copy()
        pose[2, 3] += float(ground_lift)
    return pose


def _apply_ground_lift(cloud: np.ndarray, ground_lift: float) -> np.ndarray:
    if not ground_lift or cloud.shape[0] == 0:
        return cloud
    cloud = cloud.copy()
    cloud[:, 2] -= float(ground_lift)
    return cloud


# ------------------------------------------------------------------- the stages

def load_pose_tracks(reader: BagReader, cfg: ConverterConfig,
                     report: ConversionReport,
                     clocks: Optional[clockmod.HostClocks] = None) -> Dict[str, PoseTrack]:
    """Decode every pose topic once and keep the trajectories in memory.

    Pose stamps are clock-corrected like every other stamp. Leaving them raw would
    be the subtlest form of the offset bug this converter exists to catch: the
    frame table would match a corrected cloud against an uncorrected pose, so the
    agent's data and the pose it is placed at would come from instants tens of
    milliseconds apart — a rigid position error that no downstream check would
    attribute to timing.
    """
    topics = sorted({a.pose.topic for a in cfg.active_agents if a.pose.topic})
    tracks: Dict[str, PoseTrack] = {t: PoseTrack() for t in topics}
    if not topics:
        return tracks

    host_of_topic = {a.pose.topic: a.clock_host for a in cfg.active_agents
                     if a.pose.topic}
    for topic, stamp, msg in reader.iter_messages(topics):
        transform, speed = _pose_and_speed(msg)
        if clocks is not None:
            stamp = stamp + clocks.correction_ns(host_of_topic[topic], stamp)[0]
        tracks[topic].add(stamp, transform, speed)

    for topic, track in tracks.items():
        track.finish()
        if len(track) == 0:
            raise ConversionError(f"pose topic {topic!r} carried no messages")
        if _fill_speed_from_motion(track):
            report.warnings.append(
                f"{topic}: twist is zero in every message (stationary agent, or a "
                f"driver that does not fill it in) — ego_speed derived by "
                f"differencing positions instead")
    return tracks


def resolve_world_poses(cfg: ConverterConfig, tracks: Dict[str, PoseTrack],
                        t_ns: int) -> Dict[str, Optional[Tuple[np.ndarray, float]]]:
    """``world <- base`` for every active agent at one frame time.

    The chain is ``align @ odom_T_child @ child_to_base``: the odometry message
    gives the child frame in its own odom origin, ``child_to_base`` moves that to
    the robot body, and ``align`` — the operator-supplied transform — is what puts
    all the agents into one shared world.
    """
    out: Dict[str, Optional[Tuple[np.ndarray, float]]] = {}
    for agent in cfg.active_agents:
        if agent.pose.source == "static":
            out[agent.name] = (agent.pose.static_pose.copy(), 0.0)
            continue
        track = tracks[agent.pose.topic]
        found = track.lookup(t_ns, mode=agent.pose.interpolation,
                             max_gap_ns=int(agent.pose.max_gap_ms * 1e6))
        if found is None:
            out[agent.name] = None
            continue
        odom_from_child, speed = found
        out[agent.name] = (agent.pose.align @ odom_from_child @ agent.pose.child_to_base,
                           speed)
    return out


def reconcile_clocks(reader: BagReader, cfg: ConverterConfig, indexes,
                     report: ConversionReport) -> Optional[clockmod.HostClocks]:
    """Estimate every host's clock offset before a single frame is matched.

    Returns ``None`` when reconciliation is off, in which case stamps are used as
    recorded — correct for a single-host bag and quietly wrong for any other.

    The delivery statistics come free from the index pass, which already collected
    both ``header.stamp`` and ``log_time`` per message; NTP status topics are the
    only extra decoding, and they are a few thousand small messages.
    """
    if not cfg.clock.enabled:
        return None

    reference = cfg.clock.reference_host or ego_agent(cfg).clock_host
    clocks = clockmod.HostClocks(reference, apply_corrections=(cfg.clock.mode == "correct"))

    host_of_topic = {}
    for agent in cfg.active_agents:
        host_of_topic[agent.cloud.topic] = agent.clock_host
        if agent.pose.topic:
            host_of_topic[agent.pose.topic] = agent.clock_host
        for camera in agent.cameras:
            host_of_topic[camera.topic] = agent.clock_host
    for topic, entry in indexes.items():
        host = host_of_topic.get(topic)
        if host is None or not entry.header_stamps or not entry.log_times:
            continue
        stats = clocks.delivery.setdefault(host, clockmod.DeliveryStats(host))
        for stamp, log in zip(entry.header_stamps, entry.log_times):
            stats.add(log - stamp)

    ntp_rows: Dict[str, list] = {}
    event_topics = set(cfg.clock.events_topics.values())
    topics = sorted(set(cfg.clock.ntp_topics.values()) | event_topics)
    events: Dict[str, list] = {}
    if topics:
        for topic, stamp, msg in reader.iter_messages(topics):
            if topic in event_topics:
                events.setdefault(topic, []).append((stamp, str(getattr(msg, "data", msg))))
            else:
                ntp_rows.setdefault(topic, []).append((stamp, msg))

    meta: Dict[str, dict] = {}
    for host, topic in cfg.clock.ntp_topics.items():
        track, info = clockmod.build_offset_track(
            ntp_rows.get(topic, []), offset_field=cfg.clock.offset_field,
            offset_unit=cfg.clock.offset_unit, source=topic)
        if track is None:
            report.warnings.append(
                f"clock: {host} NTP topic {topic!r} unusable ({info.get('reason')}); "
                f"fields present: {info.get('fields')}. Set clock.offset_field to "
                f"name the right one, or the host falls back to the delivery-floor "
                f"estimate.")
            continue
        clocks.ntp[host] = track
        meta[host] = info
        _health_warnings(host, info.get("health") or {}, cfg.clock.max_residual_ms,
                         _bag_start_ns(indexes), report)
        if info["unit_confidence"] in ("low", "all_zero", "none", "ambiguous"):
            report.warnings.append(
                f"clock: {host} NTP offset unit could not be inferred with "
                f"confidence ({info['unit_confidence']}). Set clock.offset_unit "
                f"explicitly — reading milliseconds as seconds inflates the "
                f"correction a thousandfold.")

    if cfg.clock.sign == "auto":
        # When the unit was declared in the config, only the sign is open; leaving
        # the scale free would let the estimator override an explicit instruction.
        scales = (1.0,) if cfg.clock.offset_unit else (1.0, 1e-3, 1e3)
        sign, scale, sign_detail = clockmod.choose_form(clocks, scales=scales)
    else:
        sign, scale = float(cfg.clock.sign), 1.0
        sign_detail = {"verdict": "sign forced by config"}
    if sign * scale != 1.0:
        for track in clocks.ntp.values():
            track.scale(sign * scale)
    if scale != 1.0:
        report.warnings.append(
            f"clock: the NTP offset unit inferred from the message did not match the "
            f"delivery floor; offsets were rescaled by {scale:g}. Set "
            f"clock.offset_unit explicitly so this is a decision and not a rescue.")
    # A near-tie on the sign only matters when the sign is about to be APPLIED.
    # In verify mode a sub-millisecond offset is a well-disciplined clock, and
    # "the two readings differ by less than the noise" is a description of
    # exactly that, not a problem.
    if sign_detail.get("verdict", "").startswith("NEAR-TIE") and cfg.clock.mode == "correct":
        report.warnings.append(
            f"clock: the NTP offset sign is a near-tie ({sign_detail}); the two "
            f"readings of the field disagree by less than the noise, so set "
            f"clock.sign explicitly rather than trusting this.")

    tolerance_ns = int(cfg.clock.cross_check_tolerance_ms * 1e6)
    midpoint = _bag_midpoint(indexes)
    report.clocks = clocks.summary(midpoint, tolerance_ns)
    report.clocks["_meta"] = {"reference_host": reference, "sign": sign,
                              "unit_rescale": scale, "sign_detail": sign_detail,
                              "ntp_fields": meta}

    t0 = min((e.header_stamps[0] for e in indexes.values() if e.header_stamps), default=0)
    t_end = max((e.header_stamps[-1] for e in indexes.values() if e.header_stamps), default=t0)
    report.clocks["_events"] = {}
    for host, topic in cfg.clock.events_topics.items():
        rows = events.get(topic, [])
        parsed = [_parse_clock_event(stamp, text, t0) for stamp, text in rows]
        report.clocks["_events"][host] = parsed
        _classify_clock_events(host, parsed, (t_end - t0) / 1e9,
                               cfg.clock.max_residual_ms, report)

    for host, entry in report.clocks.items():
        if host.startswith("_"):
            continue
        if cfg.clock.mode == "verify" and entry.get("ntp_available"):
            offset = entry["ntp"]["p95_abs_ms"]
            if offset > cfg.clock.max_residual_ms:
                report.warnings.append(
                    f"clock: {host}'s daemon reports a residual offset of {offset:.1f} ms "
                    f"(p95) — larger than clock.max_residual_ms={cfg.clock.max_residual_ms}. "
                    f"The stamps are disciplined, but not to the level assumed; every frame "
                    f"from this host carries that as clock_residual_ms. If the host was "
                    f"genuinely undisciplined, set clock.mode: correct.")
        if entry["cross_check"] == "DISAGREE":
            report.warnings.append(
                f"clock: {host}'s NTP offset and its delivery floor disagree by "
                f"{entry['cross_check_detail']['difference_ms']} ms. One of them is "
                f"wrong; do not trust this dataset's cross-agent timing until it is "
                f"resolved.")
        elif entry["estimate_source"] == "delivery_floor":
            report.warnings.append(
                f"clock: {host} publishes no NTP status, so its offset is estimated "
                f"from delivery floors alone (residual "
                f"{entry['residual_ms']:.1f} ms, carried into every frame). Adding an "
                f"NTP monitor on that host removes the last unmeasured term.")
        elif entry["estimate_source"] == "UNKNOWN":
            report.warnings.append(
                f"clock: {host}'s offset could not be estimated at all — its stamps "
                f"are used as recorded and any error appears as uniform latency on "
                f"that agent.")
    return clocks


_DELTA_RE = re.compile(r"delta\s*=\s*(-?\d+(?:\.\d+)?)\s*(ms|us|s)\b", re.I)
_MONO_RE = re.compile(r"mono\s*=\s*(\d+(?:\.\d+)?)", re.I)
_ALARM_WORDS = ("unsync", "not synchronised", "not synchronized", "lost", "no source",
                "unreachable", "leap")


def _parse_clock_event(stamp_ns: int, text: str, t0: int) -> dict:
    """One daemon event line -> {t_rel_s, text, delta_ms?, mono_s?}."""
    out = {"t_rel_s": round((stamp_ns - t0) / 1e9, 3), "text": text}
    m = _DELTA_RE.search(text)
    if m:
        value, unit = float(m.group(1)), m.group(2).lower()
        out["delta_ms"] = value * {"ms": 1.0, "us": 1e-3, "s": 1e3}[unit]
    m = _MONO_RE.search(text)
    if m:
        out["mono_s"] = float(m.group(1))
    return out


def _classify_clock_events(host: str, parsed: List[dict], bag_span_s: float,
                           max_residual_ms: float, report: ConversionReport) -> None:
    """Decide which daemon events are worth a warning.

    A monitor that logs every chrony adjustment as "STEP" produces ten of them
    for sub-millisecond slews; those are the daemon doing its job, not a
    discontinuity. What actually matters:

    * MAGNITUDE. A step is only a step if it moved the clock by more than the
      residual the frames already carry. Below max_residual_ms it is noise.
    * WHEN IT HAPPENED. A transient-local (latched) events topic replays its
      backlog the moment the recorder subscribes: N events all arriving within
      a second of each other, whose own monotonic timestamps span far more than
      the bag, are HISTORY, not events during the recording. std_msgs/String
      has no header, so their stamps are the arrival time and cannot tell you
      this; the `mono=` field can.
    """
    if not parsed:
        return
    arrivals = [e["t_rel_s"] for e in parsed]
    monos = [e["mono_s"] for e in parsed if "mono_s" in e]
    backlog = (len(parsed) >= 3
               and max(arrivals) - min(arrivals) < 1.0
               and len(monos) >= 3
               and (max(monos) - min(monos)) > 2.0 * max(bag_span_s, 1.0))
    if backlog:
        report.warnings.append(
            f"clock: {host} delivered {len(parsed)} daemon events within one second at "
            f"t={min(arrivals):.1f} s whose own monotonic timestamps span "
            f"{(max(monos) - min(monos)) / 60:.0f} minutes — a latched topic replaying its "
            f"backlog at subscription time. They predate the recording; nothing stepped "
            f"during it. Largest adjustment in the backlog: "
            f"{max((abs(e.get('delta_ms', 0.0)) for e in parsed), default=0.0):.2f} ms.")
        return
    real_steps, routine, worded = [], [], []
    for e in parsed:
        text = e["text"].lower()
        if "delta_ms" in e:
            (real_steps if abs(e["delta_ms"]) > max_residual_ms else routine).append(e)
        elif any(w in text for w in _ALARM_WORDS):
            worded.append(e)
        else:
            routine.append(e)
    for e in real_steps:
        report.warnings.append(
            f"clock: {host} clock STEPPED by {e['delta_ms']:+.1f} ms at t={e['t_rel_s']:.1f} s "
            f"({e['text']!r}). That is a discontinuity in this host's stamps that no offset "
            f"series shows; frames around it should be treated as unsynchronised.")
    for e in worded:
        report.warnings.append(
            f"clock: {host} daemon event at t={e['t_rel_s']:.1f} s: {e['text']!r} — a lost "
            f"or unreachable source means the clock was free-running from here.")
    if routine and not real_steps and not worded:
        biggest = max((abs(e.get("delta_ms", 0.0)) for e in routine), default=0.0)
        report.warnings.append(
            f"clock: {host} logged {len(routine)} routine daemon adjustment(s) during the "
            f"recording, largest {biggest:.2f} ms — below clock.max_residual_ms="
            f"{max_residual_ms:g}, so already covered by the per-frame residual.")


def _bag_start_ns(indexes) -> int:
    return min((e.header_stamps[0] for e in indexes.values() if e.header_stamps), default=0)


def _health_warnings(host: str, health: dict, max_residual_ms: float, t0_ns: int,
                     report: ConversionReport) -> None:
    """Turn the daemon's own per-sample verdicts into warnings, once each."""
    n = health.get("n", 0)
    if not n:
        return
    unsynced = health.get("unsynced_samples", 0)
    if unsynced:
        first = health.get("unsynced_stamps_ns") or [t0_ns]
        report.warnings.append(
            f"clock: {host} reported synchronized=false in {unsynced} of {n} status "
            f"samples (first at t={((first[0] - t0_ns) / 1e9):.1f} s). While unsynchronised "
            f"the clock is free-running and its stamps are not comparable to the other "
            f"hosts' — frames in those stretches should be treated as unsynchronised.")
    big = [s for s in (health.get("steps") or []) if abs(s["delta_ms"]) > max_residual_ms]
    small = [s for s in (health.get("steps") or []) if abs(s["delta_ms"]) <= max_residual_ms]
    for s in big:
        report.warnings.append(
            f"clock: {host} status flagged clock_stepped with a {s['delta_ms']:+.1f} ms "
            f"delta at t={((s['stamp_ns'] - t0_ns) / 1e9):.1f} s — a discontinuity in this "
            f"host's stamps larger than the per-frame residual covers.")
    if small and not big:
        report.warnings.append(
            f"clock: {host} status flagged {len(small)} clock step(s), all "
            f"≤ {max(abs(s['delta_ms']) for s in small):.2f} ms — routine, below "
            f"clock.max_residual_ms={max_residual_ms:g}.")
    reach = health.get("reachability_min_pct")
    if reach is not None and reach < 100:
        report.warnings.append(
            f"clock: {host} reachability dropped to {reach}% "
            f"({health.get('samples_below_full_reach', 0)} of {n} samples below 100%): the "
            f"time source was intermittently unreachable; discipline coasted on the "
            f"frequency estimate through those stretches.")
    warns = health.get("daemon_warnings") or {}
    for text, count in sorted(warns.items(), key=lambda kv: -kv[1])[:5]:
        report.warnings.append(f"clock: {host} daemon warning x{count}: {text!r}")


def _bag_midpoint(indexes) -> int:
    stamps = [s for entry in indexes.values() for s in (entry.header_stamps[:1] +
                                                        entry.header_stamps[-1:])]
    return (min(stamps) + max(stamps)) // 2 if stamps else 0


def plan(reader: BagReader, cfg: ConverterConfig,
         report: ConversionReport) -> Tuple[List[Frame], Dict[str, PoseTrack],
                                            Dict[str, dict], FrameTable,
                                            Optional[clockmod.HostClocks]]:
    """Index, synchronise and pose-resolve: everything before the first write."""
    available = reader.topics()
    wanted = cfg.topics()
    missing = [t for group in wanted.values() for t in group if t not in available]
    if missing:
        raise ConversionError(
            "these configured topics are not in the bag:\n  " +
            "\n  ".join(sorted(set(missing))) +
            "\n\nRun scripts/inspect_bag.py to list what the recording actually has.")

    cloud_topics = wanted["cloud"] + wanted["camera"]
    indexes = reader.index(cloud_topics)

    for topic, entry in indexes.items():
        if not entry.header_stamps:
            raise ConversionError(f"topic {topic!r} is present but empty")
        if entry.headerless:
            report.warnings.append(
                f"{topic}: {entry.headerless} messages had no usable header stamp; "
                f"log time was used for those")

    source = cfg.time.stamp_source
    clocks = reconcile_clocks(reader, cfg, indexes, report)
    if clocks is not None and source == "log":
        report.warnings.append(
            "time.stamp_source is 'log' while clock reconciliation is on. log_time is "
            "the recorder's clock at receipt and includes network transit, so it is "
            "already on one clock and the corrections are meaningless for it — but it "
            "also carries the transit delay into your frame times. Use 'header'.")

    def _corrected(topic: str, host: str) -> StampIndex:
        raw = indexes[topic].stamps(source)
        if clocks is None:
            return StampIndex(raw)
        return StampIndex([t + clocks.correction_ns(host, t)[0] for t in raw], raw)

    cloud_indices = {a.name: _corrected(a.cloud.topic, a.clock_host)
                     for a in cfg.active_agents}
    camera_indices = {}
    for agent in cfg.active_agents:
        for slot, camera in enumerate(agent.cameras):
            camera_indices[(agent.name, slot)] = _corrected(camera.topic,
                                                            agent.clock_host)

    master_name = cfg.time.master_agent or ego_agent(cfg).name
    master = cfg.agent_by_name(master_name)
    times = frame_times(cloud_indices[master.name].stamps,
                        rate_hz=cfg.time.rate_hz,
                        start_offset_s=cfg.time.start_offset_s,
                        duration_s=cfg.time.duration_s)
    if not times:
        raise ConversionError("no candidate frame times — check time.start_offset_s "
                              "and time.duration_s against the bag's span")

    table = build_frame_table(
        times=times,
        cloud_indices=cloud_indices,
        required={a.name: a.required for a in cfg.active_agents},
        tolerance_ns=int(cfg.time.match_tolerance_ms * 1e6),
        master=master.name,
        drop_incomplete=cfg.time.drop_incomplete_frames,
        camera_indices=camera_indices,
        stride=cfg.output.frame_stride)

    report.frames_candidate = len(times[::max(1, cfg.output.frame_stride)])
    report.dropped = dict(table.dropped)
    t0 = times[0]
    period_ns = int(np.median(np.diff(times))) if len(times) > 1 else NS // 10
    report.dropped_runs = [
        {"reason": r["reason"], "frames": r["frames"],
         "t_start_s": round((r["start_ns"] - t0) / 1e9, 1),
         "t_end_s": round((r["end_ns"] - t0) / 1e9, 1)}
        for r in table.dropped_runs(max_gap_ns=2 * period_ns) if r["frames"] >= 3]
    report.sync = reuse_statistics(table)
    residuals: Dict[str, float] = {}
    for agent in cfg.active_agents:
        index = cloud_indices[agent.name]
        entry = report.sync.setdefault(agent.name, {})
        entry["source_rate_hz"] = round(index.rate_hz(), 3)
        # Half a publication period: the tightest this stream can ever be matched,
        # whatever the tolerance says. Reported per agent so a tolerance that is
        # structurally unreachable is visible before the frames vanish.
        entry["half_period_ms"] = round(index.half_period_ns() / 1e6, 3)
        if clocks is not None:
            residual, residual_source = clocks.residual_ns(agent.clock_host)
            residuals[agent.name] = residual / 1e6
            entry["clock_residual_ms"] = round(residual / 1e6, 4)
            entry["clock_source"] = clocks.correction_ns(agent.clock_host, times[0])[1]
    report.tightness = tightness_curve(
        table, [a.name for a in cfg.active_agents if a.required], residuals)
    worst_floor = max(((cloud_indices[a.name].half_period_ns() / 1e6, a.name)
                       for a in cfg.active_agents if a.required),
                      default=(0.0, None))
    report.tightness["structural_floor_ms"] = round(worst_floor[0], 3)
    report.tightness["structural_floor_agent"] = worst_floor[1]
    if cfg.time.match_tolerance_ms < worst_floor[0]:
        report.warnings.append(
            f"time.match_tolerance_ms={cfg.time.match_tolerance_ms} is below the "
            f"structural floor of {worst_floor[0]:.1f} ms set by "
            f"{worst_floor[1]!r} at {cloud_indices[worst_floor[1]].rate_hz():.1f} Hz. "
            f"Half a publication period is the best nearest-neighbour matching can "
            f"do, so this tolerance drops frames for a reason no processing can fix — "
            f"raise it, or record that stream faster.")

    tracks = load_pose_tracks(reader, cfg, report, clocks)

    kept: List[Frame] = []
    frame_poses: Dict[str, dict] = {}
    pose_failures: Dict[str, int] = {}
    for frame in table.frames:
        poses = resolve_world_poses(cfg, tracks, frame.t_ns)
        blocking = [a.name for a in cfg.active_agents
                    if a.required and poses.get(a.name) is None]
        if blocking:
            for name in blocking:
                pose_failures[name] = pose_failures.get(name, 0) + 1
            continue
        frame.index = len(kept)
        kept.append(frame)
        frame_poses[frame.key] = poses

    for name, count in pose_failures.items():
        key = f"no_pose:{name}"
        report.dropped[key] = report.dropped.get(key, 0) + count

    if not kept:
        raise ConversionError(
            "every candidate frame was dropped. Most common causes: "
            "time.match_tolerance_ms too tight for the slowest agent, or a pose "
            "topic that does not span the bag (see the drop reasons above).")

    report.pose_stats = _pose_statistics(cfg, frame_poses, kept)
    _check_expected_starts(cfg, report)
    return kept, tracks, frame_poses, table, clocks


def _check_expected_starts(cfg: ConverterConfig, report: ConversionReport) -> None:
    """Refuse a pose source that starts somewhere other than where the operator
    said it would.

    The typical way this fires: the trajectory was republished at a body frame
    (a base_link on the floor under a sensor mast) while the config treats it as
    the camera optical frame. Every extrinsic in the config is then measured from
    the wrong point — a metre out vertically — and the dataset converts, validates
    and looks fine.
    """
    bad = []
    for agent in cfg.active_agents:
        expected = agent.pose.expected_start
        stats = report.pose_stats.get(agent.name)
        if expected is None or stats is None:
            continue
        start = np.asarray(stats["start_m"], dtype=np.float64)
        gap = start - np.asarray(expected, dtype=np.float64)
        dist = float(np.linalg.norm(gap))
        stats["expected_start_m"] = [round(float(v), 3) for v in expected]
        stats["start_gap_m"] = round(dist, 3)
        if dist > agent.pose.expected_start_tolerance_m:
            bad.append((agent, start, expected, gap, dist))
    if not bad:
        return
    lines = []
    for agent, start, expected, gap, dist in bad:
        lines.append(
            f"  {agent.name}: base starts at {np.round(start, 3).tolist()} but "
            f"pose.expected_start is {list(expected)} — {dist:.2f} m apart "
            f"(dx {gap[0]:+.2f}, dy {gap[1]:+.2f}, dz {gap[2]:+.2f}). ")
        if abs(gap[2]) > 0.5 and abs(gap[2]) > 3 * float(np.hypot(gap[0], gap[1])):
            lines.append(
                f"     Almost entirely vertical: the {agent.pose.topic!r} poses are "
                f"most likely of a BODY frame under the sensor (a base_link at floor "
                f"level), not the camera optical frame this config assumes. Either "
                f"republish them at the optical frame, or set pose.child_to_base to "
                f"the body -> optical transform and re-express every extrinsic from "
                f"that body frame.")
    raise ConversionError(
        "pose source does not start where it should:\n" + "\n".join(lines) +
        "\n\nThis is a frame check, not a precision check: a mismatch this size "
        "means the poses and the extrinsics are measured from different points. "
        "Fix the frame, or remove pose.expected_start if the anchor itself is wrong.")


def _pose_statistics(cfg: ConverterConfig, frame_poses: Dict[str, dict],
                     frames: List[Frame]) -> Dict[str, dict]:
    """Trajectory extent and worst inter-frame jump, per agent.

    A large jump usually means the alignment transform is fine but the SLAM
    relocalised — worth seeing before trusting any cross-agent geometry.
    """
    stats: Dict[str, dict] = {}
    t0 = frames[0].t_ns if frames else 0
    for agent in cfg.active_agents:
        positions, jumps, jump_times, jump_dts = [], [], [], []
        previous, previous_t = None, None
        for frame in frames:
            entry = frame_poses[frame.key].get(agent.name)
            if entry is None:
                continue
            position = entry[0][:3, 3]
            positions.append(position)
            if previous is not None:
                jumps.append(float(np.linalg.norm(position - previous)))
                jump_times.append((frame.t_ns - t0) / 1e9)
                jump_dts.append(max((frame.t_ns - previous_t) / 1e9, 1e-6))
            previous, previous_t = position, frame.t_ns
        if not positions:
            continue
        positions = np.asarray(positions)
        stats[agent.name] = {
            "frames": int(positions.shape[0]),
            "start_m": [round(float(v), 3) for v in positions[0]],
            "end_m": [round(float(v), 3) for v in positions[-1]],
            "path_length_m": round(float(np.sum(jumps)), 3) if jumps else 0.0,
            "extent_m": [round(float(v), 3)
                         for v in (positions.max(axis=0) - positions.min(axis=0))],
            "max_step_m": round(float(np.max(jumps)), 4) if jumps else 0.0,
            # The three fastest inter-frame moves, as SPEED. A raw step is
            # misleading: consecutive KEPT frames can be seconds apart when the
            # frames between them were dropped, so a 0.26 m step over 1.9 s is a
            # walk (0.14 m/s), not a jump. A real pose discontinuity is a large
            # step over a normal frame interval. `gap` flags a dt well above the
            # frame period, i.e. a step measured across dropped frames.
            "largest_steps": _fastest_moves(jumps, jump_times, jump_dts),
            "z_span_m": round(float(positions[:, 2].max() - positions[:, 2].min()), 3),
        }
    return stats


def _fastest_moves(steps, times, dts, keep: int = 3) -> List[dict]:
    if not steps:
        return []
    nominal = float(np.median(dts))
    rows = [{"t_s": round(t, 1), "step_m": round(j, 3), "dt_s": round(d, 3),
             "speed_mps": round(j / d, 3), "gap": bool(d > 1.5 * nominal)}
            for j, t, d in zip(steps, times, dts)]
    return sorted(rows, key=lambda r: -r["speed_mps"])[:keep]


def assign_scenarios(frames: List[Frame], cfg: ConverterConfig) -> List[Tuple[str, List[Frame]]]:
    """Chop the frame list into OPV2V scenario folders."""
    size = cfg.output.frames_per_scenario
    if size <= 0 or size >= len(frames):
        return [(cfg.output.scenario_name, frames)]
    chunks = []
    for start in range(0, len(frames), size):
        block = frames[start:start + size]
        chunks.append((f"{cfg.output.scenario_name}_{start // size:03d}", block))
    return chunks


def write_frame_yamls(cfg: ConverterConfig, scenarios, frame_poses: Dict[str, dict],
                      report: ConversionReport,
                      clocks: Optional[clockmod.HostClocks] = None) -> Dict[str, str]:
    """Write every agent's per-frame yaml; returns frame key -> scenario dir."""
    agent_objects = {
        a.name: {"object_id": a.obj.object_id, "extent": a.obj.extent,
                 "center": a.obj.center, "extrinsic": a.obj.extrinsic}
        for a in cfg.active_agents if a.obj.emit}
    statics = load_static_labels(cfg, report)

    split_root = os.path.join(cfg.output.root, cfg.output.split)
    frame_dirs: Dict[str, str] = {}

    for scenario_name, frames in scenarios:
        scenario_dir = os.path.join(split_root, scenario_name)
        for local_index, frame in enumerate(frames):
            key = f"{local_index:06d}"
            frame_dirs[frame.key] = scenario_dir
            frame.local_key = key                     # consumed by the write pass
            poses = frame_poses[frame.key]
            world_poses = {name: entry[0] if entry else None
                           for name, entry in poses.items()}

            for agent in cfg.active_agents:
                entry = poses.get(agent.name)
                if entry is None:
                    continue
                world_from_base, speed = entry
                lidar_pose = matrix_to_opencood_pose(
                    # frame_extrinsic(), not extrinsic: when the points were
                    # rotated to body axes the pose must carry the same rotation.
                    _sensor_pose(world_from_base, agent.cloud.frame_extrinsic(),
                                 agent.cloud.ground_lift))
                body_pose = matrix_to_opencood_pose(world_from_base)

                params = {
                    "ego_speed": float(speed * 3.6),          # OPV2V stores km/h
                    "lidar_pose": lidar_pose,
                    "true_ego_pos": body_pose,
                    "predicted_ego_pos": body_pose,
                    "plan_trajectory": [],
                    "vehicles": labels.merge_external_labels(
                        labels.vehicles_for_viewer(
                            world_poses, agent_objects, viewer=agent.name,
                            include_self=cfg.output.include_self_in_vehicles),
                        statics),
                }
                for slot, camera in enumerate(agent.cameras):
                    params[f"camera{slot}"] = _camera_block(
                        camera, world_from_base, agent, cfg)

                # Provenance: the real sensor time behind this synthetic frame key.
                params["ros_stamp_ns"] = int(frame.cloud_stamps.get(agent.name,
                                                                    frame.t_ns))
                params["ros_frame_stamp_ns"] = int(frame.t_ns)
                params["source_agent"] = agent.name
                params["ros_sync"] = _sync_block(cfg, agent, frame, clocks)

                write_frame_yaml(
                    os.path.join(scenario_dir, str(agent.cav_id), f"{key}.yaml"),
                    params)
    return frame_dirs


def load_static_labels(cfg: ConverterConfig, report: ConversionReport) -> List[dict]:
    """Static world-frame objects, read once and written into every frame.

    They are world-frame facts, so nothing here is per-frame: the same list goes
    to every agent, and `labels.merge_external_labels` folds it into each
    `vehicles` dict beside whatever agent-derived boxes exist.
    """
    path = cfg.output.labels_file
    if not path:
        return []
    with open(path) as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        raise ConversionError(f"{path}: expected a list of label objects")
    out = []
    for item in raw:
        missing = [k for k in ("id", "location", "extent") if k not in item]
        if missing:
            raise ConversionError(f"{path}: a label is missing {missing}")
        out.append({"id": int(item["id"]), "location": [float(v) for v in item["location"]],
                    "extent": [float(v) for v in item["extent"]],
                    "angle": [float(v) for v in item.get("angle", (0.0, 0.0, 0.0))],
                    "center": [float(v) for v in item.get("center", (0.0, 0.0, 0.0))]})
    names = ", ".join(str(item.get("name", item["id"])) for item in raw)
    report.warnings.append(
        f"labels: {len(out)} static object(s) from {os.path.basename(path)} ({names}) "
        f"written into every frame of every agent. They are world-frame boxes: verify "
        f"them with `label_static.py --check` before trusting any number computed "
        f"against them.")
    return out



def _sync_block(cfg: ConverterConfig, agent: AgentConfig, frame: Frame,
                clocks: Optional[clockmod.HostClocks]) -> dict:
    """How synchronous this agent's contribution to this frame actually is.

    Written per frame, not just aggregated into the conversion report, because an
    aggregate mean hides exactly the case that matters: a handful of frames where
    one agent's message is far older than the rest. Downstream, this block is what
    lets an experiment exclude or stratify by realised asynchrony instead of
    assuming it away — and this study's own results (100 ms of latency costing more
    than 90% packet loss) are the reason that distinction is not academic.

    ``clock_residual_ms`` is the part that correcting the host clocks could not
    remove; ``total_ms`` is the sum, i.e. the honest bound on how stale this
    agent's data is relative to the frame time.
    """
    skew_ms = frame.skew_ns.get(agent.name, 0) / 1e6
    residual_ms, residual_source = (0.0, "disabled")
    correction_source = "disabled"
    if clocks is not None:
        residual, residual_source = clocks.residual_ns(agent.clock_host)
        residual_ms = residual / 1e6
        correction_source = clocks.correction_ns(agent.clock_host, frame.t_ns)[1]
    block = {
        "host": agent.clock_host,
        "cloud_dt_ms": round(skew_ms, 4),
        "clock_residual_ms": round(residual_ms, 4),
        "clock_correction_source": correction_source,
        "clock_residual_source": residual_source,
        "total_ms": round(abs(skew_ms) + residual_ms, 4),
        "pose_interpolation": agent.pose.interpolation,
    }
    cameras = {f"camera{slot}": round(frame.camera_skew_ns[(agent.name, slot)] / 1e6, 4)
               for slot in range(len(agent.cameras))
               if (agent.name, slot) in frame.camera_skew_ns}
    if cameras:
        block["camera_dt_ms"] = cameras
    return block


def _camera_block(camera, world_from_base: np.ndarray, agent: AgentConfig,
                  cfg: ConverterConfig) -> dict:
    """OPV2V-shaped camera metadata (``cords`` / ``extrinsic`` / ``intrinsic``).

    Stock OpenCOOD's LiDAR pipeline never reads these; they are written so
    camera-capable forks and visual debugging have what they need.
    """
    world_from_camera = world_from_base @ camera.extrinsic
    lidar_pose = _sensor_pose(world_from_base, agent.cloud.frame_extrinsic(),
                              agent.cloud.ground_lift)
    camera_to_lidar = invert(lidar_pose) @ world_from_camera
    return {
        "cords": matrix_to_opencood_pose(world_from_camera),
        "extrinsic": [[float(v) for v in row] for row in camera_to_lidar],
        "intrinsic": camera.intrinsic or [[0.0, 0.0, 0.0]] * 3,
    }


def write_clouds(reader: BagReader, cfg: ConverterConfig, frames: List[Frame],
                 frame_dirs: Dict[str, str], camera_infos: Dict[str, object],
                 report: ConversionReport,
                 tracks: Optional[Dict[str, PoseTrack]] = None,
                 progress: Optional[Callable[[int, int], None]] = None) -> None:
    """The streaming write pass: decode only the selected messages."""
    # topic -> {stamp: [(agent, frame)]}, so one message can serve several frames
    selection: Dict[str, Dict[int, List[Tuple[AgentConfig, Frame, int]]]] = {}
    for frame in frames:
        for agent in cfg.active_agents:
            stamp = frame.cloud_stamps.get(agent.name)
            if stamp is not None:
                selection.setdefault(agent.cloud.topic, {}) \
                    .setdefault(stamp, []).append((agent, frame, -1))
        for (agent_name, slot), stamp in frame.camera_stamps.items():
            agent = cfg.agent_by_name(agent_name)
            selection.setdefault(agent.cameras[slot].topic, {}) \
                .setdefault(stamp, []).append((agent, frame, slot))

    stamp_sets = {topic: set(stamps) for topic, stamps in selection.items()}
    totals: Dict[str, List[int]] = {a.name: [] for a in cfg.active_agents}
    written = 0
    total_targets = sum(len(v) for stamps in selection.values() for v in stamps.values())

    for topic, stamp, msg in reader.iter_selected(stamp_sets):
        for agent, frame, slot in selection[topic].get(stamp, ()):
            target_dir = os.path.join(frame_dirs[frame.key], str(agent.cav_id))
            key = getattr(frame, "local_key", frame.key)
            if slot < 0:
                cloud = _build_cloud(agent, msg, camera_infos, tracks, frame, stamp,
                                     report)
                count = write_pcd(os.path.join(target_dir, f"{key}.pcd"), cloud)
                totals[agent.name].append(count)
            else:
                image = image_to_array(msg)
                if image is not None:
                    write_png(os.path.join(target_dir, f"{key}_camera{slot}.png"), image)
            written += 1
            if progress and written % 200 == 0:
                progress(written, total_targets)

    for name, counts in totals.items():
        if not counts:
            continue
        report.points_per_agent[name] = {
            "frames": len(counts),
            "mean_points": round(float(np.mean(counts)), 1),
            "min_points": int(np.min(counts)),
            "max_points": int(np.max(counts)),
        }
        if int(np.min(counts)) == 0:
            report.warnings.append(
                f"{name}: at least one frame produced an empty cloud — check "
                f"cloud.range_filter and, for depth clouds, min/max_depth")


def _build_cloud(agent: AgentConfig, msg, camera_infos: Dict[str, object],
                 tracks: Optional[Dict[str, PoseTrack]] = None,
                 frame: Optional[Frame] = None, stamp_ns: Optional[int] = None,
                 report: Optional[ConversionReport] = None) -> np.ndarray:
    """Message -> (N, 4) cloud in the agent's *sensor* frame, ready to write."""
    cloud_cfg = agent.cloud
    if cloud_cfg.kind == "pointcloud2":
        if cloud_cfg.deskew and tracks is not None and frame is not None:
            cloud, offsets = cloud_from_pointcloud2(msg, cloud_cfg.intensity,
                                                    cloud_cfg.point_time_field)
            cloud = _deskew(agent, cloud, offsets, stamp_ns, frame, tracks, report)
        else:
            cloud = cloud_from_pointcloud2(msg, cloud_cfg.intensity)
    else:
        info = camera_infos.get(cloud_cfg.camera_info_topic)
        if info is None:
            raise ConversionError(
                f"agent {agent.name}: no CameraInfo seen on "
                f"{cloud_cfg.camera_info_topic!r}, cannot reproject depth")
        cloud = cloud_from_depth_image(msg, info, cloud_cfg)

    cloud = apply_range_filter(cloud, cloud_cfg.range_filter)
    cloud = subsample(cloud, cloud_cfg.max_points)
    return _apply_ground_lift(cloud, cloud_cfg.ground_lift)


def _deskew(agent: AgentConfig, cloud: np.ndarray, offsets, stamp_ns: Optional[int],
            frame: Frame, tracks: Dict[str, PoseTrack],
            report: Optional[ConversionReport]) -> np.ndarray:
    """Motion-compensate one sweep to its frame's reference time.

    The target time is the frame time, not the message stamp, so the correction
    absorbs the agent's selection skew as well as the sweep's own duration: after
    it, the cloud is what this sensor would have seen had it observed the whole
    scene instantaneously at ``frame.t_ns``. The pose track is the agent's raw
    odometry, so the operator-supplied ``align`` never enters and a wrong alignment
    cannot corrupt the result.

    A cloud whose pose track cannot cover the sweep is left alone and counted;
    ``deskew.skipped`` in the conversion report says how often that happened, which
    is the signal that a SLAM dropout overlaps the recording.
    """
    track = tracks.get(agent.pose.topic) if agent.pose.topic else None
    stats = None if report is None else report.deskew.setdefault(
        agent.name, {"applied": 0, "skipped": 0, "reasons": {}})
    if track is None or agent.pose.source == "static":
        if stats is not None:
            stats["skipped"] += 1
            stats["reasons"]["no pose track (static agent?)"] = \
                stats["reasons"].get("no pose track (static agent?)", 0) + 1
        return cloud
    # Deskew runs AFTER any optical->body rotation, so the frame it corrects in
    # is the frame the points are already in. Identical to `extrinsic` for a
    # PointCloud2, which is the only kind that carries per-point times today.
    sensor_from_base = agent.pose.child_to_base @ agent.cloud.frame_extrinsic()
    out, info = deskew_cloud(
        cloud, offsets, int(stamp_ns if stamp_ns is not None else frame.t_ns),
        int(frame.t_ns), track, sensor_from_base,
        max_gap_ns=int(agent.pose.max_gap_ms * 1e6),
        buckets=agent.cloud.deskew_buckets)
    if stats is not None:
        if info["applied"]:
            stats["applied"] += 1
            stats["sweep_span_ms"] = info["sweep_span_ms"]
        else:
            stats["skipped"] += 1
            reason = info.get("reason", "unknown")
            stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
    return out


def load_camera_infos(reader: BagReader, cfg: ConverterConfig) -> Dict[str, object]:
    """First ``CameraInfo`` per topic — intrinsics do not change within a bag."""
    topics = cfg.topics()["camera_info"]
    infos: Dict[str, object] = {}
    if not topics:
        return infos
    remaining = set(topics)
    for topic, stamp, msg in reader.iter_messages(topics):
        if topic in remaining:
            infos[topic] = msg
            remaining.discard(topic)
            if not remaining:
                break
    return infos


def convert(cfg: ConverterConfig, overwrite: bool = False,
            progress: Optional[Callable[[int, int], None]] = None) -> ConversionReport:
    """Run the full conversion described by ``cfg``."""
    started = time.time()
    report = ConversionReport(bag=cfg.bag,
                              output=os.path.join(cfg.output.root, cfg.output.split))

    reader = BagReader(cfg.bag, cfg.time.stamp_source)
    frames, tracks, frame_poses, table, clocks = plan(reader, cfg, report)

    scenarios = assign_scenarios(frames, cfg)
    report.scenarios = [name for name, _ in scenarios]

    split_root = os.path.join(cfg.output.root, cfg.output.split)
    for scenario_name, _ in scenarios:
        scenario_dir = os.path.join(split_root, scenario_name)
        if os.path.isdir(scenario_dir):
            if not overwrite:
                raise ConversionError(
                    f"{scenario_dir} already exists. Re-run with --overwrite to "
                    f"replace it, or change output.scenario_name.")
            shutil.rmtree(scenario_dir)
        for agent in cfg.active_agents:
            os.makedirs(os.path.join(scenario_dir, str(agent.cav_id)), exist_ok=True)

    camera_infos = load_camera_infos(reader, cfg)
    for agent in cfg.active_agents:
        for slot, camera in enumerate(agent.cameras):
            if camera.intrinsic is None and camera.camera_info_topic in camera_infos:
                fx, fy, cx, cy = camera_intrinsics(camera_infos[camera.camera_info_topic])
                camera.intrinsic = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]

    frame_dirs = write_frame_yamls(cfg, scenarios, frame_poses, report, clocks)
    write_clouds(reader, cfg, frames, frame_dirs, camera_infos, report, tracks, progress)

    report.frames_written = len(frames)
    report.duration_s = time.time() - started

    if cfg.output.write_provenance:
        provenance = os.path.join(split_root, "conversion_report.json")
        os.makedirs(os.path.dirname(provenance), exist_ok=True)
        with open(provenance, "w") as handle:
            json.dump(report.to_dict(), handle, indent=2)
    return report
