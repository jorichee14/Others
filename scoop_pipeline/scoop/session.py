# -*- coding: utf-8 -*-
"""
One pass of the rig -- every machine's bags of one sequence -- and the check
that they hold what the machines were told to record.

    data/raw/<date>/<pass>/          e.g. 20260924/mapping_A
        mobile_1/   <prefix>_<pass>_..._<date>_mobile_1/   + .svo2
        mobile_2/   <prefix>_<pass>_..._<date>_mobile_2/
        infra_1/    ...
    data/work/<date>/<pass>/
        mobile_1/   <bag>_retimed/  <svo>_zed/  <svo>_zed_right/  (process_recording.py)
        <prefix>_<pass>_..._<date>_merged/                        (merge_session.py)

What a machine records is ``configs/record.yaml`` -- the robots' own file,
which ``record.sh`` reads. A machine's bags are found by their name
(``..._<machine>``, ``/`` as ``_``, a repeat's ``_r2`` ignored); a new machine
(mobile_2, infra_N) needs only its entry there.

A machine contributes its processed bags where they exist (mobile_1: the
retimed bag and the ZED bags, whose topics are the recorded ones renamed plus
the decoded ones -- see :func:`processed_topics`), else its raw bags.
"""
from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from . import bag, recording, replay
from .bag import BagError

__all__ = ["RecordPlan", "Unit", "TopicCheck", "find_units", "processed_topics",
           "check_topics", "session_name"]

DEFAULT_PLAN = recording.ROOT / "configs" / "record.yaml"


class RecordPlan:
    """``record.yaml``: which topics each machine records, and (its ``check``
    section) which of them a pass mode does not."""

    def __init__(self, path=None):
        self.path = Path(path or DEFAULT_PLAN).expanduser()
        with open(self.path) as fh:
            doc = yaml.safe_load(fh) or {}
        self.machines: Dict[str, List[str]] = {
            m: list((spec or {}).get("topics") or [])
            for m, spec in (doc.get("machines") or {}).items()}
        self.tf_recorded_by = doc.get("tf_recorded_by")
        chk = doc.get("check") or {}
        self.skip: List[str] = list(chk.get("skip") or [])
        self.modes: Dict[str, List[str]] = {
            m: list((spec or {}).get("skip") or []) for m, spec in (chk.get("modes") or {}).items()}
        # recorded, but publishing only when something happens: no messages is fine
        self.may_be_empty: List[str] = list(chk.get("may_be_empty") or [])

    def mode_of(self, *names: str) -> Optional[str]:
        """The first mode named as a word in ``names`` (the pass folder first,
        then bag names): mapping_A -> mapping, ..._coop_... -> coop."""
        for name in names:
            words = re.split(r"[_\W]+", name.lower())
            for m in self.modes:
                if m.lower() in words:
                    return m
        return None

    def skipped(self, topic: str, mode: Optional[str] = None) -> bool:
        pats = self.skip + (self.modes.get(mode, []) if mode else [])
        return any(fnmatch.fnmatchcase(topic, p) for p in pats)

    def topics(self, machine: str, mode: Optional[str] = None) -> List[str]:
        """What ``record.sh <machine>`` records (/tf and /tf_static on the
        tf machine), minus what the check skips in ``mode``."""
        t = list(self.machines[machine])
        if machine == self.tf_recorded_by:
            t += ["/tf", "/tf_static"]
        return [x for x in t if not self.skipped(x, mode)]

    def machine_of(self, bag_name: str) -> Optional[str]:
        """The machine a bag belongs to, from its name's end."""
        stem = re.sub(r"_r\d+$", "", bag_name)
        for m in sorted(self.machines, key=len, reverse=True):   # mobile_1_sniffer first
            if stem.endswith("_" + m.replace("/", "_")):
                return m
        return None


@dataclass
class TopicCheck:
    counts: Dict[str, int]                 # topic -> messages, over all the bags
    missing: List[str]                     # expected, not in any bag
    empty: List[str]                       # expected, there, no messages
    extra: List[str]                       # in a bag, not expected
    quiet: List[str] = field(default_factory=list)   # no messages, allowed (may_be_empty)

    @property
    def ok(self) -> bool:
        return not self.missing and not self.empty


@dataclass
class Unit:
    """One machine's contribution to a pass."""
    machine: str
    raw_dir: Path
    bags: List[Path]                       # what goes into the merge
    expected: List[str]                    # topics those bags must have
    processed: bool = False
    raw_bags: List[Path] = field(default_factory=list)
    may_be_empty: List[str] = field(default_factory=list)

    def check(self) -> TopicCheck:
        return check_topics(self.bags, self.expected, self.may_be_empty)


def check_topics(bags, expected, may_be_empty=()) -> TopicCheck:
    """Which of ``expected`` are missing or empty in ``bags`` (counts from
    each bag's metadata.yaml / MCAP summary; nothing is read). A topic
    matching ``may_be_empty`` must be there but may have no messages."""
    counts: Dict[str, int] = {}
    for b in bags:
        for t, info in bag.open_bag(b).topics().items():
            counts[t] = counts.get(t, 0) + int(info.count)
    exp = list(dict.fromkeys(expected))
    zero = [t for t in exp if t in counts and counts[t] == 0]
    quiet = [t for t in zero if any(fnmatch.fnmatchcase(t, p) for p in may_be_empty)]
    return TopicCheck(counts,
                      missing=[t for t in exp if t not in counts],
                      empty=[t for t in zero if t not in quiet],
                      extra=sorted(set(counts) - set(exp)),
                      quiet=quiet)


def processed_topics(raw_topics: List[str], settings: dict,
                     rec: recording.Recording) -> List[str]:
    """The topics ``process_recording.py`` turns a recording's ``raw_topics``
    into: the retimed bag has every recorded topic (renamed by the ouster
    remaps, minus ``exclude``) plus the decoded points/imu/metadata, and each
    ZED bag its topics, renamed."""
    o = settings["ouster"]
    remap = replay.parse_remaps(o.get("remap"))
    exclude = set(o.get("exclude") or ())
    copied = raw_topics if o.get("copy_all") else [t for t in raw_topics if t in remap]
    out = [remap.get(t, t) for t in copied if t not in exclude]
    ns = o.get("out_ns", "/mobile_1/ouster")
    out += [f"{ns}/points", f"{ns}/metadata"] + ([f"{ns}/imu"] if o.get("imu", True) else [])
    if rec.svo is not None:
        z = settings["zed"]
        for topics in recording.zed_bags(z).values():
            rm = recording.zed_args(z, topics)["remap"]
            out += [rm.get(t, t) for t in topics]
    return list(dict.fromkeys(out))


def _bag_folders(folder: Path) -> List[Path]:
    return sorted(p for p in folder.iterdir()
                  if p.is_dir() and (p / "metadata.yaml").is_file())


def find_units(session_raw, plan: RecordPlan, settings: dict,
               work_root=None, mode: Optional[str] = None, log=print) -> List[Unit]:
    """Every machine's bags below ``session_raw`` (a pass folder in raw/),
    checked for the topics of ``mode`` (see :meth:`RecordPlan.mode_of`)."""
    session_raw = Path(session_raw).expanduser().resolve()
    bag_dirs = [p.parent for p in session_raw.rglob("metadata.yaml")]
    folders = sorted({b.parent for b in bag_dirs})          # the folders holding bags
    units: List[Unit] = []
    for folder in folders:
        raw_bags = _bag_folders(folder)
        by_machine: Dict[str, List[Path]] = {}
        for b in raw_bags:
            m = plan.machine_of(b.name) or (folder.name if folder.name in plan.machines else None)
            if m is None:
                log(f"note: {b} belongs to no machine in {plan.path.name}; left out")
                continue
            by_machine.setdefault(m, []).append(b)
        for m, bags in by_machine.items():
            units.append(_unit(m, folder, bags, plan, settings, work_root, mode))
    if not units:
        raise BagError(f"no bags of any machine in {plan.path.name} below {session_raw}")
    return units


def _unit(machine, folder, raw_bags, plan, settings, work_root, mode) -> Unit:
    """Processed bags if the folder's recording was processed, else raw."""
    expected = plan.topics(machine, mode)
    try:
        rec = recording.find_recording(folder, work_root,
                                       settings["ouster"].get("packets_ns", "/ouster"),
                                       recording.zed_bags(settings["zed"]))
    except BagError:
        rec = None
    if rec is not None and rec.bag in raw_bags and rec.done("retimed"):
        bags = [rec.step("retimed")]
        if rec.svo is not None:
            bags += [rec.zed_bag(n) for n in rec.zed_bags
                     if (rec.zed_bag(n) / "metadata.yaml").is_file()]
        others = [b for b in raw_bags if b != rec.bag]          # other bags there, as is
        return Unit(machine, folder, bags + others,
                    processed_topics(expected, settings, rec), True, raw_bags,
                    plan.may_be_empty)
    return Unit(machine, folder, list(raw_bags), expected, False, raw_bags, plan.may_be_empty)


def session_name(units: List[Unit], plan: RecordPlan) -> str:
    """``<prefix>_<pass>_..._<date>``: the bags' names without the machine."""
    stems = []
    for u in units:
        for b in u.raw_bags:
            stem = re.sub(r"_r\d+$", "", b.name)
            suffix = "_" + u.machine.replace("/", "_")
            stems.append(stem[:-len(suffix)] if stem.endswith(suffix) else stem)
    if not stems:
        raise BagError("no raw bags to name the session after")
    return max(set(stems), key=stems.count)
