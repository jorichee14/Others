"""Configuration, and the refusals.

The rule this harness inherits from `rosbag_to_opv2v`: a value that cannot be
recovered from the data is `null` in the config and the loader REFUSES rather
than substituting a default. A SLAM benchmark has three such values, and every
one of them produces a run that completes, validates and is wrong:

  1. the sensor-to-reference extrinsic. A method tracks the sensor it was given;
     the reference describes a different frame on the same robot. Leave it at
     identity and the error you measure is the lever arm, modulated by how much
     the platform rotated.
  2. the reference's own uncertainty. Without it there is no way to say whether
     a 4 mm ATE difference between two methods is a result.
  3. the evaluation volume and resolution for map metrics. Both are choices that
     change the ranking, so neither gets a silent default.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import yaml

from . import se3
from .trajectory import Trajectory

__all__ = ["DatasetConfig", "MethodConfig", "load_dataset", "load_method",
           "extrinsic_matrix", "to_reference_frame"]


class ConfigError(ValueError):
    """A refusal. The message says what to supply and where to get it."""


def _req(d: dict, key: str, where: str):
    if key not in d:
        raise ConfigError(f"{where}: missing required key '{key}'")
    v = d[key]
    if v is None:
        raise ConfigError(f"{where}: '{key}' is null. This value cannot be guessed — "
                          f"see the comment above it in the config.")
    return v


@dataclasses.dataclass
class DatasetConfig:
    name: str
    raw: dict
    path: Path

    @property
    def reference(self) -> dict:
        return self.raw["reference"]

    @property
    def eval(self) -> dict:
        return self.raw["eval"]

    def stream(self, key: str) -> dict:
        s = self.raw.get("streams", {})
        if key not in s:
            raise ConfigError(f"stream {key!r} is not declared in {self.path.name}. "
                              f"Declared: {sorted(s)}")
        return s[key]

    def agent_reference(self, agent: str) -> dict:
        a = self.reference.get("agents", {})
        if agent not in a:
            raise ConfigError(
                f"no reference trajectory declared for agent {agent!r}. "
                f"Declared: {sorted(a)}. An agent with no reference can still be "
                f"run (tier-3 self-consistency), but not scored for ATE.")
        return a[agent]

    def anchors(self, agent: str | None = None) -> list[dict]:
        """Anchors usable for tier 2, resolved for one agent.

        Windows and board observations are PER AGENT: the two platforms dwell
        at different boards at different times, and on coop2 `mobile_1` cannot
        use `anchor_b` at all (it never gets closer than 2.48 m, where the
        15 mm markers are half a pixel per bit). So `windows_by_agent` and
        `observed_poses_by_agent` are the declared form and this resolves them
        down to the flat `window` / `observed_poses` absolute_check expects.

        `observed_poses_by_agent` names a TUM of board-derived camera poses in
        the map frame — coop2's mapping pipeline emits these directly. The file
        is loaded here so absolute_check stays free of file I/O.
        """
        out = []
        for a in self.reference.get("anchors", []):
            a = dict(a)
            if agent is not None:
                wba = a.get("windows_by_agent")
                if isinstance(wba, dict):
                    a["window"] = wba.get(agent)
                obp = a.get("observed_poses_by_agent")
                if isinstance(obp, dict) and obp.get(agent):
                    from .trajectory import load_tum
                    path = Path(obp[agent]).expanduser()
                    if not path.is_absolute():
                        path = self.path.parent / path
                    # A missing or unreadable file is a CONFIG problem, and it
                    # must surface as one row of tier 2 saying so -- not as a
                    # traceback out of the middle of a scoring run that has
                    # already spent minutes on tiers 1 and 3.
                    try:
                        a["observed_poses"] = load_tum(path)
                    except (OSError, ValueError) as e:
                        a["observed_poses_error"] = f"{path}: {e}"
            if a.get("window") in (None, [None, None]):
                continue          # dwell not timed, or this board is unusable here
            out.append(a)
        return out

    def reference_uncertainty_m(self) -> float:
        return float(_req(self.reference, "uncertainty_m", "reference"))

    def volume(self) -> dict:
        v = _req(self.eval, "volume", "eval")
        return {"min": _req(v, "min", "eval.volume"), "max": _req(v, "max", "eval.volume")}


@dataclasses.dataclass
class MethodConfig:
    name: str
    raw: dict
    path: Path

    @property
    def modality(self) -> list[str]:
        return list(self.raw.get("modality", []))

    @property
    def needs_imu(self) -> bool:
        return bool(self.raw.get("needs_imu", False))

    @property
    def outputs(self) -> list[str]:
        return list(self.raw.get("outputs", ["trajectory"]))

    @property
    def streams(self) -> list[str]:
        """The streams this method is run on.

        A list, not one per agent: KISS-ICP runs on both of mobile_1's geometry
        streams, and that pair — same estimator, same platform, same motion, one
        reference, two sensors — is what carries the modality comparison. Keying
        on the agent would have made it unrepresentable.
        """
        st = self.raw.get("streams")
        if not st:
            raise ConfigError(f"{self.path.name}: 'streams' is required — the list of "
                              f"stream keys from the dataset config this method runs on")
        return list(st)

    def sensor_extrinsic(self, dataset: DatasetConfig, stream_key: str) -> np.ndarray:
        """`reference_frame <- sensor_frame` for this method on this stream.

        From the method config when it overrides, otherwise from the dataset's
        stream declaration. Refuses when neither supplies it.
        """
        over = self.raw.get("extrinsic_by_stream", {})
        if stream_key in over:
            return extrinsic_matrix(over[stream_key],
                                    f"{self.path.name}:extrinsic_by_stream.{stream_key}")
        e = dataset.stream(stream_key).get("reference_frame_from_sensor")
        if e is None:
            raise ConfigError(
                f"method {self.name!r} on stream {stream_key!r}: no "
                f"'reference_frame_from_sensor' extrinsic. The method reports poses of "
                f"its own sensor frame and the reference reports a different frame on "
                f"the same robot; without this transform the ATE is the lever arm.")
        return extrinsic_matrix(e, f"{dataset.path.name}:streams")


def extrinsic_matrix(d: dict, where: str = "extrinsic") -> np.ndarray:
    """A config extrinsic block -> 4x4. Angles are degrees, Z-Y-X, matching the
    `extrinsic:` blocks in rosbag_to_opv2v configs so the two can be copied
    between projects without re-derivation."""
    for k in ("x", "y", "z", "roll", "pitch", "yaw"):
        if d.get(k) is None:
            raise ConfigError(f"{where}: extrinsic component '{k}' is null")
    return se3.pose_from_rpy(d["x"], d["y"], d["z"], d["roll"], d["pitch"], d["yaw"])


def to_reference_frame(traj: Trajectory, reference_frame_from_sensor: np.ndarray,
                       ref_frame_name: str = "") -> Trajectory:
    """Re-express `world <- sensor` poses as `world <- reference_frame`.

    world<-ref = (world<-sensor) @ (sensor<-ref), and (sensor<-ref) is the
    inverse of the extrinsic, which is declared as ref<-sensor to match the
    `base_link -> sensor` convention of the converter configs.
    """
    out = traj.transform_right(se3.invert(reference_frame_from_sensor))
    out.frame = ref_frame_name or traj.frame
    return out


def load_dataset(path: str | Path) -> DatasetConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    for key in ("dataset", "reference", "eval"):
        if key not in raw:
            raise ConfigError(f"{path.name}: missing top-level '{key}' block")
    cfg = DatasetConfig(name=_req(raw["dataset"], "name", "dataset"), raw=raw, path=path)
    cfg.reference_uncertainty_m()       # refuse at load time, not at report time
    return cfg


def load_method(path: str | Path) -> MethodConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    name = _req(raw, "name", path.name)
    if "modality" not in raw:
        raise ConfigError(f"{path.name}: 'modality' is required — the modalities this "
                          f"method ACCEPTS. Which comparison block a row lands in comes "
                          f"from the stream it ran on, not from this list.")
    return MethodConfig(name=name, raw=raw, path=path)
