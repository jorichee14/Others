# -*- coding: utf-8 -*-
"""
Walks the stage list for one unit: decides what is current, runs what is not.

Each stage runs in its OWN Python process. When it exits the operating system
takes back everything it held -- a 100M-point cloud, a KD-tree, CuPy's memory
pool -- before the next stage starts, and a crash in one stage cannot leave
another half-initialised. Output is shown live and appended to the stage's
``log.txt``.

A stage is current when its ``stamp.json`` exists and its config view, its
script and its input fingerprints are unchanged. Small outputs are content
hashed, so re-running a stage that writes an identical trajectory does not
cascade; large ones (clouds) are compared by size and mtime, so a rebuilt map
does re-run what reads it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from . import stamp as stampmod
from .config import ROOT, digest, stage_view
from .layout import PASS, SEQ, SITE, Layout, Unit
from .stages import (ORDER, Stage, as_lists, resolve_inputs, resolve_outputs,
                     stages_for)

CTX_FILE = ".ctx.json"


@dataclass
class Item:
    stage: Stage
    unit: Unit
    inputs: Dict[str, List[str]]
    outputs: Dict[str, List[str]]
    work: Path
    config_view: Dict
    config_sha: str
    script_sha: Optional[str]
    reason: Optional[str] = None           # None = current

    @property
    def label(self) -> str:
        return f"{self.unit.rel}:{self.stage.name}"


def plan(unit: Unit, cfg: Dict, layout: Layout) -> List[Item]:
    """Every (stage, unit) the request covers, in execution order."""
    if unit.level == SITE:
        pairs = [(s, unit.at_pass(p)) for p in layout.pass_ids() for s in stages_for(PASS)]
        pairs += [(s, unit) for s in stages_for(SITE)]
    else:
        pairs = [(s, unit) for s in stages_for(unit.level)]
    items = []
    for stage, u in pairs:
        view = stage_view(cfg, stage.name, stage.uses)
        items.append(Item(
            stage=stage, unit=u,
            inputs=as_lists(resolve_inputs(stage, u, layout)),
            outputs=as_lists(resolve_outputs(stage, u, layout)),
            work=layout.work(u, stage.name),
            config_view=view, config_sha=digest(view),
            script_sha=stampmod.file_sha(stage.script)))
    return items


def assess(item: Item) -> Optional[str]:
    if not item.stage.implemented:
        return f"not implemented (phase {item.stage.phase})"
    item.reason = stampmod.why_stale(
        stampmod.read(item.work), config_sha=item.config_sha,
        script_sha=item.script_sha, inputs=item.inputs, outputs=item.outputs)
    return item.reason


def select(items: List[Item], only=(), start=None, until=None) -> List[Item]:
    rank = {n: i for i, n in enumerate(ORDER)}
    for n in list(only) + [x for x in (start, until) if x]:
        if n not in rank:
            raise SystemExit(f"unknown stage {n!r}; stages: {', '.join(ORDER)}")
    lo = rank[start] if start else -1
    hi = rank[until] if until else len(ORDER)
    return [it for it in items
            if lo <= rank[it.stage.name] <= hi and (not only or it.stage.name in only)]


def status(items: List[Item], out=print):
    w = max(len(it.label) for it in items) if items else 10
    for it in items:
        reason = assess(it)
        st = stampmod.read(it.work)
        when = ""
        if st:
            when = f"   last run {st.get('started_utc', '?')}, {st.get('duration_s', '?')} s"
        state = "ok" if reason is None else reason
        out(f"  {it.label:{w}s}  {state}{when if reason is None else ''}")


def _run_process(item: Item, cfg: Dict) -> int:
    item.work.mkdir(parents=True, exist_ok=True)
    ctx = {"stage": item.stage.name, "unit": item.unit.rel, "script": str(item.stage.script),
           "work": str(item.work), "cfg": cfg,
           "inputs": {k: v if (len(v) != 1 or item.stage.inputs[k].startswith("each:"))
                      else v[0] for k, v in item.inputs.items()},
           "outputs": {k: v[0] for k, v in item.outputs.items()}}
    ctx_path = item.work / CTX_FILE
    ctx_path.write_text(json.dumps(ctx, indent=2, default=str))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    with open(item.work / "log.txt", "a") as log:
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')}  {item.label} =====\n")
        proc = subprocess.Popen([sys.executable, "-u", "-m", "scoop.stage_main", str(ctx_path)],
                                cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write("    " + line)
            log.write(line)
        return proc.wait()


def execute(items: List[Item], cfg: Dict, force: Iterable[str] = (),
            dry_run: bool = False, out=print) -> int:
    """Run what is stale (or forced). Stops at the first failure or the first
    stage that is not implemented yet. Returns a process exit code."""
    force = set(force)
    for it in items:
        reason = assess(it)
        if not it.stage.implemented:
            out(f"[stop] {it.label}: {reason}")
            return 0 if dry_run else 3
        if reason is None and it.stage.name not in force:
            out(f"[skip] {it.label}: up to date")
            continue
        why = "forced" if it.stage.name in force else reason
        if dry_run:
            out(f"[would run] {it.label}: {why}")
            continue
        missing = [p for p in stampmod.all_paths(it.inputs) if not Path(p).exists()]
        if missing:
            out(f"[fail] {it.label}: input missing: {missing[0]}")
            return 2
        out(f"[run ] {it.label}: {why}")
        old = it.work / stampmod.STAMP
        if old.exists():
            old.unlink()                     # a failed rerun must not look done
        started = time.time()
        code = _run_process(it, cfg)
        if code != 0:
            out(f"[fail] {it.label}: exit code {code}; log {it.work / 'log.txt'}")
            return 1
        missing = [p for p in stampmod.all_paths(it.outputs) if not Path(p).exists()]
        if missing:
            out(f"[fail] {it.label}: finished but did not write {missing[0]}")
            return 1
        stampmod.write(it.work, stage=it.stage.name, unit=it.unit.rel,
                       config_view=it.config_view, config_sha=it.config_sha,
                       script_sha=it.script_sha, inputs=it.inputs, outputs=it.outputs,
                       started=started, repo=ROOT)
        out(f"[done] {it.label}: {time.time() - started:.1f} s")
    return 0
