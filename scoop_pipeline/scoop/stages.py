# -*- coding: utf-8 -*-
"""
The stage list: what runs, in which order, reading what, writing what.

This file is the pipeline. ``run.py`` only walks it. A stage is one script
under ``stages/<level>/<name>.py`` exposing ``run(ctx)``; it reads
``ctx.inputs``, writes ``ctx.outputs``, and takes its parameters from
``ctx.params`` (= ``cfg["stages"][name]``) and the sections it lists in
``uses``.

Input references (resolved per unit by :func:`resolve`):

    @bag, @traj           passes.<P>.bag / .traj of the site config (pass stages)
    @raw                  the unit's raw directory (sequence stages)
    <stage>/<file>        another stage's output, same unit
    ref:<stage>/<file>    ... of the reference mapping pass
    each:<stage>/<file>   ... of every mapping pass (a list)
    site:<stage>/<file>   ... of the site-level stage
    release:<path>        below the unit's release directory

Outputs are a bare file name (in the stage's own work directory) or a
``release:`` path. Order within a level is execution order; a stage may only
reference stages before it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .config import ROOT
from .layout import PASS, SEQ, SITE, Layout, LayoutError, Unit

PathOrList = Union[Path, List[Path]]


@dataclass(frozen=True)
class Stage:
    name: str
    level: str
    inputs: Dict[str, str]
    outputs: Dict[str, str]
    uses: Tuple[str, ...] = ()
    phase: int = 1
    doc: str = ""
    path: Optional[Path] = None             # tests only; default below

    @property
    def script(self) -> Path:
        return self.path or ROOT / "stages" / self.level / f"{self.name}.py"

    @property
    def implemented(self) -> bool:
        return self.script.is_file()


# --------------------------------------------------------------------------- #
# Phase 1: site pipeline. Pass stages run once per mapping recording (A, B);
# site stages once per site, after every pass.
# --------------------------------------------------------------------------- #
PASS_STAGES: List[Stage] = [
    Stage("s01_seed_map", PASS,
          inputs={"bag": "@bag", "traj": "@traj"},
          outputs={"denoised": "denoised.pcd"},
          uses=("sensor", "build_map"),
          doc="01_build_map up to denoise, on the seed (GLIM) trajectory"),
    Stage("s02_refine", PASS,
          inputs={"bag": "@bag", "traj": "@traj", "map": "s01_seed_map/denoised.pcd"},
          outputs={"traj": "traj_refined.tum", "rounds": "rounds.json"},
          uses=("sensor",),
          doc="01a: register every scan to the map, N rounds"),
    Stage("s03_final_map", PASS,
          inputs={"bag": "@bag", "traj": "s02_refine/traj_refined.tum"},
          outputs={"map": "map.pcd"},
          uses=("sensor", "build_map"),
          doc="01_build_map in full on the refined trajectory"),
    Stage("s04_cut", PASS,
          inputs={"map": "s03_final_map/map.pcd"},
          outputs={"map": "cut.pcd", "hist": "z_hist.json"},
          doc="02: floor/ceiling cut; bounds are chosen in the notebook"),
    Stage("s05_anchor", PASS,
          inputs={"bag": "@bag", "traj": "s02_refine/traj_refined.tum",
                  "map": "s04_cut/cut.pcd"},
          outputs={"map": "anchored.pcd", "frame": "frame.json"},
          uses=("sensor", "boards"),
          doc="03: boards in the map frame, cloud anchored to the first board"),
]

SITE_STAGES: List[Stage] = [
    Stage("s06_reference", SITE,
          inputs={"map": "ref:s05_anchor/anchored.pcd",
                  "frame": "ref:s05_anchor/frame.json"},
          outputs={"cache": "refcache.npz",
                   "map": "release:reference/map.pcd",
                   "boards": "release:reference/boards.json"},
          doc="freeze the reference pass: release files + the registration "
              "cache (points, normals, plane table) every sequence loads"),
    Stage("s07_validate", SITE,
          inputs={"maps": "each:s05_anchor/anchored.pcd",
                  "frames": "each:s05_anchor/frame.json"},
          outputs={"report": "validate.json"},
          doc="pass A vs pass B: thickness, placement, scale (Table gt)"),
]

# --------------------------------------------------------------------------- #
# Phase 2: sequence pipeline, against the frozen site reference. Inputs and
# outputs are provisional until each stage is implemented.
# --------------------------------------------------------------------------- #
SEQ_STAGES: List[Stage] = [
    Stage("q01_lidar_loc", SEQ, phase=2,
          inputs={"raw": "@raw", "ref": "site:s06_reference/refcache.npz"},
          outputs={"traj": "release:poses/mobile_1.tum"},
          doc="mobile_1 scan-to-map registration (01a register, one pass)"),
    Stage("q02_rgbd_loc", SEQ, phase=2,
          inputs={"raw": "@raw", "ref": "site:s06_reference/refcache.npz"},
          outputs={"traj": "release:poses/mobile_2.tum"},
          doc="mobile_2 depth registration + board constraints"),
    Stage("q03_infra", SEQ, phase=2,
          inputs={"raw": "@raw"},
          outputs={"poses": "release:poses/infra.json"},
          doc="infrastructure poses through the board chain"),
    Stage("q04_board_check", SEQ, phase=2,
          inputs={"raw": "@raw"},
          outputs={"report": "board_check.json"},
          doc="held-out board residuals, pose consistency"),
    Stage("q05_geometry", SEQ, phase=2,
          inputs={"raw": "@raw"},
          outputs={"geometry": "release:geometry"},
          doc="per-frame collaboration geometry (_geom.yaml)"),
    Stage("q06_extract", SEQ, phase=2,
          inputs={"raw": "@raw"},
          outputs={"done": "extract.json"},
          doc="OPV2V-style extraction through rosbag_to_opv2v"),
]

ALL = {s.name: s for s in PASS_STAGES + SITE_STAGES + SEQ_STAGES}
ORDER = [s.name for s in PASS_STAGES + SITE_STAGES + SEQ_STAGES]


def stages_for(level: str) -> List[Stage]:
    return {PASS: PASS_STAGES, SITE: SITE_STAGES, SEQ: SEQ_STAGES}[level]


def resolve(ref: str, unit: Unit, layout: Layout, stage: Stage) -> PathOrList:
    """One input/output reference -> path(s) for this unit."""
    if ref.startswith("release:"):
        return layout.release(unit) / ref[len("release:"):]
    if ref.startswith("@"):
        key = ref[1:]
        if key == "raw":
            return layout.raw(unit)
        if unit.level != PASS:
            raise LayoutError(f"{stage.name}: {ref} only exists for mapping passes")
        return layout.pass_input(unit, key)
    scope, _, rest = ref.rpartition(":")
    if "/" not in rest:
        raise LayoutError(f"{stage.name}: bad reference {ref!r}")
    other, fname = rest.split("/", 1)
    if scope == "":
        return layout.work(unit, other) / fname
    if scope == "ref":
        return layout.work(unit.at_pass(layout.reference_pass()), other) / fname
    if scope == "each":
        return [layout.work(unit.at_pass(p), other) / fname for p in layout.pass_ids()]
    if scope == "site":
        return layout.work(unit.site_unit(), other) / fname
    raise LayoutError(f"{stage.name}: unknown scope {scope!r} in {ref!r}")


def resolve_outputs(stage: Stage, unit: Unit, layout: Layout) -> Dict[str, Path]:
    out = {}
    for name, ref in stage.outputs.items():
        out[name] = (layout.release(unit) / ref[len("release:"):]
                     if ref.startswith("release:") else layout.work(unit, stage.name) / ref)
    return out


def resolve_inputs(stage: Stage, unit: Unit, layout: Layout) -> Dict[str, PathOrList]:
    return {name: resolve(ref, unit, layout, stage) for name, ref in stage.inputs.items()}


def as_lists(d: Dict[str, PathOrList]) -> Dict[str, List[str]]:
    return {k: [str(p) for p in (v if isinstance(v, list) else [v])] for k, v in d.items()}
