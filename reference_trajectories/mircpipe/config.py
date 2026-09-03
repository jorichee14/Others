"""One config file, one run context.

The config keeps the shape you already have - a `dataset` block plus one
block per stage - and adds two things a multi-stage pipeline needs:

  "paths":  { "raw_root": "...", "out_root": "map_stages_20260828_outputs",
              "results_root": "results" }
  "runs":   { "coop2":    { "bag": "<path>" },
              "mapping2": { "bag": "<path>" } }

Every stage block may name a `run` instead of repeating a bag path, and
every relative path in the config is resolved against the config file's own
directory, so a pipeline can be run from anywhere.

`Ctx` is passed to every stage. It carries the resolved paths and the caches
that are expensive to build (the reference map, the parquet cache), so a
five-stage run loads the map once instead of five times.
"""
import json
import os


class Cfg(dict):
    """The config dict with path resolution and stage lookup."""

    def __init__(self, raw, base_dir):
        super().__init__(raw)
        self.base_dir = base_dir
        self.paths = raw.get("paths", {})

    # ---------------------------------------------------------------- paths
    def path(self, p, root=None):
        """Resolve a config path: absolute stays, relative is taken against
        `root` (a key in "paths") or the config file's directory."""
        if p is None:
            return None
        p = os.path.expanduser(str(p))
        if os.path.isabs(p):
            return p
        if root and self.paths.get(root):
            base = os.path.expanduser(self.paths[root])
            if not os.path.isabs(base):
                base = os.path.join(self.base_dir, base)
            return os.path.normpath(os.path.join(base, p))
        return os.path.normpath(os.path.join(self.base_dir, p))

    def out_root(self):
        return self.path(self.paths.get("out_root", "."))

    def results_root(self):
        return self.path(self.paths.get("results_root", "results"))

    # --------------------------------------------------------------- stages
    def stage(self, name, required=True):
        """The block for a stage. Accepts the bare name ("08_reference") or
        the short key ("08", "reference")."""
        if name in self:
            return self[name]
        for k in self:
            if not isinstance(k, str):
                continue
            if k == name or k.split("_", 1)[0] == name or k.endswith("_" + name):
                return self[k]
        if required:
            raise SystemExit("no '%s' block in the config; blocks present: %s"
                             % (name, sorted(k for k in self if isinstance(k, str))))
        return None

    def run_bag(self, name):
        """Bag path of a named run from the "runs" block."""
        runs = self.get("runs", {})
        if name not in runs:
            raise SystemExit("run '%s' is not in the config's \"runs\" block (have %s)"
                             % (name, sorted(runs)))
        return self.path(runs[name]["bag"], root="raw_root")

    def bag_of(self, block, run=None):
        """The bag a stage should read: its own "bag", else its "run", else
        the run given on the command line, else the dataset bag."""
        if isinstance(block, dict):
            if block.get("bag"):
                return self.path(block["bag"], root="raw_root")
            if block.get("run"):
                return self.run_bag(block["run"])
        if run:
            return self.run_bag(run)
        ds = self.get("dataset", {})
        if ds.get("bag"):
            return self.path(ds["bag"], root="raw_root")
        raise SystemExit("no bag: give the stage a \"bag\" or \"run\", or pass --run")


def load_config(path):
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(path):
        raise SystemExit("config not found: %s" % path)
    with open(path) as f:
        raw = json.load(f)
    return Cfg(raw, os.path.dirname(path))


class Ctx:
    """Shared state for one pipeline invocation.

    Stages take (cfg, ctx) and use ctx for anything worth building once:
      ctx.reference(path)   the frozen map as a mapref.Reference
      ctx.parquet(bag)      the extracted parquet cache directory
      ctx.out(*parts)       a path under the run's output directory
      ctx.result(*parts)    a path under results/<run>/
    """

    def __init__(self, cfg, run=None, out_dir=None, verbose=True):
        self.cfg = cfg
        self.run = run
        self.verbose = verbose
        self._out_dir = out_dir
        self._ref = {}
        self._parquet = {}

    # ----------------------------------------------------------- locations
    def out(self, *parts):
        base = self._out_dir or self.cfg.out_root()
        p = os.path.join(base, *[str(x) for x in parts])
        os.makedirs(os.path.dirname(p) if os.path.splitext(p)[1] else p, exist_ok=True)
        return p

    def result(self, *parts):
        p = os.path.join(self.cfg.results_root(), self.run or "run",
                         *[str(x) for x in parts])
        os.makedirs(os.path.dirname(p) if os.path.splitext(p)[1] else p, exist_ok=True)
        return p

    def log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    # -------------------------------------------------------------- caches
    def reference(self, map_path, voxel=0.05, plane_voxel=0.4):
        """The frozen map, built once per path even across stages."""
        from . import mapref, bag
        key = (os.path.abspath(map_path), voxel, plane_voxel)
        if key not in self._ref:
            self.log("  loading reference map %s" % map_path)
            self._ref[key] = mapref.Reference(bag.read_map_xyz(map_path),
                                              voxel=voxel, plane_voxel=plane_voxel)
        return self._ref[key]

    def parquet(self, bag_path, topics=None, refresh=False):
        """Directory of the parquet cache for a bag, extracting on first use."""
        from . import cache
        key = os.path.abspath(bag_path)
        if refresh or key not in self._parquet:
            self._parquet[key] = cache.ensure(bag_path, self.out("cache"),
                                              topics=topics, refresh=refresh,
                                              printer=self.log)
        return self._parquet[key]
