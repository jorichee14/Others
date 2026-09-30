# -*- coding: utf-8 -*-
"""
Entry point of one stage process: ``python -m scoop.stage_main <ctx.json>``.

The runner writes the context, this module loads the stage script and calls
its ``run(ctx)``. A stage script therefore never parses arguments or looks
for its own config; it gets everything in ``ctx``:

    ctx.stage      stage name
    ctx.unit       Unit (site / pass / session / seq)
    ctx.cfg        the full merged config
    ctx.params     cfg["stages"][stage]  (what the stamp hashes, with `uses`)
    ctx.inputs     name -> Path  (or list of Path for `each:` references)
    ctx.outputs    name -> Path  (parent directories already exist)
    ctx.work       the stage's own work directory, for anything else it keeps
    ctx.layout     Layout, for the rare stage that needs another path
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from .layout import Layout, parse_unit


class Ctx:
    def __init__(self, d):
        self.stage = d["stage"]
        self.unit = parse_unit(d["unit"])
        self.cfg = d["cfg"]
        self.params = self.cfg.get("stages", {}).get(self.stage, {}) or {}
        self.work = Path(d["work"])
        self.layout = Layout(self.cfg)
        self.inputs = {k: [Path(p) for p in v] if isinstance(v, list) else Path(v)
                       for k, v in d["inputs"].items()}
        self.outputs = {k: Path(v) for k, v in d["outputs"].items()}
        for p in self.outputs.values():
            p.parent.mkdir(parents=True, exist_ok=True)

    def __repr__(self):
        return f"Ctx({self.stage} @ {self.unit})"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    d = json.loads(Path(argv[0]).read_text())
    spec = importlib.util.spec_from_file_location(f"stage_{d['stage']}", d["script"])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "run"):
        raise SystemExit(f"{d['script']} has no run(ctx)")
    mod.run(Ctx(d))


if __name__ == "__main__":
    main()
