# -*- coding: utf-8 -*-
"""
``stamp.json``: the record every stage leaves next to its outputs.

It answers two questions. *Is this output current?* -- the stage's config
view, its script and its inputs are compared with what produced the output.
*Where did this number come from?* -- the git revision, the full config view,
and the input fingerprints are kept, so a table in the paper can be traced to
the run that made it (the provenance block 03_anchor already wrote, for every
stage).

Input fingerprints are size + mtime for large files and directories (a bag
or a 100M-point cloud is not re-hashed on every status check) and a content
hash for small ones (configs, trajectories, JSON), where an edit that keeps
the size is plausible.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

STAMP = "stamp.json"
SMALL = 16 << 20                       # content-hash files up to 16 MiB


def _file_fp(p: Path) -> Dict:
    st = p.stat()
    fp = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
    if st.st_size <= SMALL:
        fp = {"size": st.st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()[:16]}
    return fp


def fingerprint(path: Path) -> Optional[Dict]:
    """Fingerprint of a file or a directory (all files below it), None if
    the path does not exist."""
    p = Path(path)
    if p.is_file():
        return _file_fp(p)
    if p.is_dir():
        files = sorted(f for f in p.rglob("*") if f.is_file())
        h = hashlib.sha256()
        total = 0
        for f in files:
            st = f.stat()
            total += st.st_size
            h.update(f"{f.relative_to(p)}|{st.st_size}|{st.st_mtime_ns}\n".encode())
        return {"files": len(files), "size": total, "tree": h.hexdigest()[:16]}
    return None


def git_rev(cwd: Path) -> Dict:
    def run(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=cwd,
                                           stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    rev = run("rev-parse", "--short", "HEAD")
    dirty = run("status", "--porcelain", "--", ".")
    return {"rev": rev, "dirty": bool(dirty) if dirty is not None else None}


def file_sha(path: Path) -> Optional[str]:
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.is_file() else None


def write(stage_dir: Path, *, stage: str, unit: str, config_view: Dict,
          config_sha: str, script_sha: Optional[str], inputs: Dict[str, List[str]],
          outputs: Dict[str, List[str]], started: float, repo: Path) -> Path:
    """Write the stamp AFTER the stage succeeded; its absence means 'not done'."""
    rec = {
        "stage": stage, "unit": unit,
        "config_sha": config_sha, "script_sha": script_sha,
        "config": config_view,
        "inputs": {name: {p: fingerprint(Path(p)) for p in paths}
                   for name, paths in inputs.items()},
        "outputs": {name: {p: fingerprint(Path(p)) for p in paths}
                    for name, paths in outputs.items()},
        "git": git_rev(repo),
        "host": socket.gethostname(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "duration_s": round(time.time() - started, 1),
    }
    path = Path(stage_dir) / STAMP
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=2, default=str))
    os.replace(tmp, path)
    return path


def read(stage_dir: Path) -> Optional[Dict]:
    p = Path(stage_dir) / STAMP
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def why_stale(stamp: Optional[Dict], *, config_sha: str, script_sha: Optional[str],
              inputs: Dict[str, List[str]], outputs: Dict[str, List[str]]) -> Optional[str]:
    """None if current, else a short human reason (the first one found)."""
    if stamp is None:
        return "never run"
    for name, paths in outputs.items():
        for p in paths:
            if not Path(p).exists():
                return f"output {name} missing ({p})"
    if stamp.get("config_sha") != config_sha:
        return "config changed"
    if script_sha and stamp.get("script_sha") != script_sha:
        return "stage script changed"
    old = stamp.get("inputs", {})
    for name, paths in inputs.items():
        prev = old.get(name, {})
        if sorted(prev) != sorted(paths):
            return f"input {name} is a different set of paths"
        for p in paths:
            now = fingerprint(Path(p))
            if now is None:
                return f"input {name} missing ({p})"
            if now != prev.get(p):
                return f"input {name} changed ({Path(p).name})"
    return None


def all_paths(d: Dict[str, Iterable]) -> List[str]:
    return [p for ps in d.values() for p in ps]
