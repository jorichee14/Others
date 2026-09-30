# -*- coding: utf-8 -*-
"""
Where everything lives on disk.

    <data_root>/
      raw/      site_1/mapping_A/...                  bags (read-only)
                site_1/session_02/seq_05/units/...
      work/     site_1/mapping_A/<stage>/             intermediates, deletable
                site_1/_site/<stage>/                 site-level stages
                site_1/session_02/seq_05/<stage>/
      release/  site_1/reference/                     exactly the paper's
                site_1/session_02/seq_05/...          release tree

A stage writes only into its own work directory, plus the release paths it
declares. Nothing else creates files, so deleting a work directory is always
safe and ``release/`` is the dataset, not a copy of it.

Three kinds of unit, one per stage level:

    pass  site_1/mapping_A              one mapping recording of a site
    site  site_1                        once per site (reference, validation)
    seq   site_1/session_02/seq_05      one dataset sequence
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

PASS, SITE, SEQ = "pass", "site", "seq"

_SITE = re.compile(r"^site_\w+$")
_SESSION = re.compile(r"^session_\w+$")
_SEQ = re.compile(r"^seq_\w+$")


class LayoutError(ValueError):
    pass


@dataclass(frozen=True)
class Unit:
    site: str
    pass_id: Optional[str] = None           # "A", "B" for mapping passes
    session: Optional[str] = None
    seq: Optional[str] = None

    @property
    def level(self) -> str:
        if self.seq:
            return SEQ
        return PASS if self.pass_id else SITE

    @property
    def rel(self) -> str:
        """Path of this unit below raw/, work/ and release/."""
        if self.seq:
            return f"{self.site}/{self.session}/{self.seq}"
        if self.pass_id:
            return f"{self.site}/mapping_{self.pass_id}"
        return self.site

    def __str__(self):
        return self.rel

    def at_pass(self, pass_id: str) -> "Unit":
        return Unit(self.site, pass_id=pass_id)

    def site_unit(self) -> "Unit":
        return Unit(self.site)


def parse_unit(spec: str) -> Unit:
    """'site_1' | 'site_1/mapping_A' | 'site_1/session_02/seq_05' -> Unit."""
    parts = [p for p in str(spec).strip("/").split("/") if p]
    if not parts or not _SITE.match(parts[0]):
        raise LayoutError(f"{spec!r}: expected site_X[/mapping_P | /session_YY/seq_ZZ]")
    if len(parts) == 1:
        return Unit(parts[0])
    if len(parts) == 2 and parts[1].startswith("mapping_"):
        return Unit(parts[0], pass_id=parts[1][len("mapping_"):])
    if len(parts) == 3 and _SESSION.match(parts[1]) and _SEQ.match(parts[2]):
        return Unit(parts[0], session=parts[1], seq=parts[2])
    raise LayoutError(f"{spec!r}: expected site_X[/mapping_P | /session_YY/seq_ZZ]")


class Layout:
    """Path resolution for one config (one data_root)."""

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.root = Path(cfg["data_root"])

    # top-level trees
    def raw(self, unit: Unit) -> Path:
        return self.root / "raw" / unit.rel

    def work(self, unit: Unit, stage: str = "") -> Path:
        base = self.root / "work" / (unit.rel if unit.level != SITE
                                     else f"{unit.site}/_site")
        return base / stage if stage else base

    def release(self, unit: Unit) -> Path:
        if unit.level == SEQ:
            return self.root / "release" / unit.rel
        return self.root / "release" / unit.site

    # per-pass inputs named in the site config
    def pass_input(self, unit: Unit, key: str) -> Path:
        """``passes.<P>.<key>`` of the site config; relative paths are taken
        from data_root, so a site file stays valid when the disk moves."""
        passes = self.cfg.get("passes", {})
        if unit.pass_id not in passes:
            raise LayoutError(f"{unit}: pass {unit.pass_id!r} not in config "
                              f"(have {sorted(passes)})")
        val = passes[unit.pass_id].get(key)
        if not val:
            raise LayoutError(f"{unit}: passes.{unit.pass_id}.{key} is not set")
        p = Path(str(val)).expanduser()
        return p if p.is_absolute() else self.root / p

    def pass_ids(self):
        return sorted(self.cfg.get("passes", {}))

    def reference_pass(self) -> str:
        ref = self.cfg.get("reference_pass")
        ids = self.pass_ids()
        if ref is None and len(ids) == 1:
            return ids[0]
        if ref not in ids:
            raise LayoutError(f"reference_pass {ref!r} is not one of the passes {ids}")
        return ref
