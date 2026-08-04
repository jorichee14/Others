#!/usr/bin/env python
"""Phase 0 gate checks. Run on the GPU machine, inside the `opencood` conda env.

Usage:
    python verify_phase0.py --stage env
    python verify_phase0.py --stage dataset --dataset-root ~/cpfa/data/OPV2V
    python verify_phase0.py --stage checkpoints --checkpoint-root ~/cpfa/checkpoints
    python verify_phase0.py --stage all --dataset-root ... --checkpoint-root ...

Exit code 0 = gate passed. Paste the printed report back into the session.
"""
import argparse
import os
import sys

PASS = "PASS"
FAIL = "FAIL"
_results = []


def check(name, fn):
    try:
        detail = fn()
        _results.append((PASS, name, detail or ""))
    except Exception as e:  # noqa: BLE001 - report every failure, never crash the gate
        _results.append((FAIL, name, "%s: %s" % (type(e).__name__, e)))


def stage_env():
    def torch_check():
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("torch imported but cuda_available=False")
        return "torch %s, cuda %s, device %s" % (
            torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))

    def numpy_check():
        import numpy
        major, minor = (int(x) for x in numpy.__version__.split(".")[:2])
        if (major, minor) >= (1, 24):
            raise RuntimeError("numpy %s >= 1.24 breaks OpenCOOD (np.float); pip install 'numpy<1.24'"
                               % numpy.__version__)
        return "numpy %s" % numpy.__version__

    def spconv_check():
        # Import the compiled core, not just the top-level package — a cumm/spconv binary
        # mismatch only surfaces when spconv.core_cc loads.
        from spconv.utils import Point2VoxelCPU3d  # noqa: F401
        import spconv
        return "spconv %s (compiled core OK)" % getattr(spconv, "__version__", "unknown")

    def cumm_check():
        from importlib import metadata
        names = sorted({(d.metadata["Name"] or "") for d in metadata.distributions()})
        cumms = [n for n in names if n.lower().startswith("cumm")]
        if len(cumms) > 1:
            raise RuntimeError(
                "conflicting cumm packages installed (%s) — they overwrite each other's "
                "files and break spconv; keep only the CUDA-specific one (cumm-cuXXX)"
                % ", ".join(cumms))
        return ", ".join(cumms) if cumms else "no cumm dist found"

    def opencood_check():
        import opencood
        return "opencood at %s" % os.path.dirname(opencood.__file__)

    check("torch + CUDA", torch_check)
    check("numpy < 1.24", numpy_check)
    check("spconv compiled core", spconv_check)
    check("single cumm package", cumm_check)
    check("opencood import", opencood_check)


def stage_dataset(root):
    root = os.path.expanduser(root)
    test_dir = os.path.join(root, "test")

    def structure():
        if not os.path.isdir(test_dir):
            raise RuntimeError("no test/ directory under %s" % root)
        scenarios = sorted(d for d in os.listdir(test_dir)
                           if os.path.isdir(os.path.join(test_dir, d)))
        if not scenarios:
            raise RuntimeError("test/ contains no scenario folders")
        n_pairs, problems = 0, []
        for sc in scenarios:
            sc_path = os.path.join(test_dir, sc)
            cavs = [d for d in os.listdir(sc_path)
                    if os.path.isdir(os.path.join(sc_path, d))]
            if not cavs:
                problems.append("%s: no CAV folders" % sc)
                continue
            for cav in cavs:
                files = os.listdir(os.path.join(sc_path, cav))
                yamls = {f[:-5] for f in files if f.endswith(".yaml")}
                pcds = {f[:-4] for f in files if f.endswith(".pcd")}
                if yamls != pcds:
                    problems.append("%s/%s: %d yaml vs %d pcd (mismatch)"
                                    % (sc, cav, len(yamls), len(pcds)))
                n_pairs += len(yamls & pcds)
        if problems:
            raise RuntimeError("; ".join(problems[:5])
                               + (" (+%d more)" % (len(problems) - 5) if len(problems) > 5 else ""))
        return "%d scenarios, %d frame-CAV pairs" % (len(scenarios), n_pairs)

    check("OPV2V test split structure", structure)


def stage_checkpoints(root):
    root = os.path.expanduser(root)

    def one(ckpt_dir):
        def load():
            import re
            import torch
            cfg = os.path.join(ckpt_dir, "config.yaml")
            if not os.path.isfile(cfg):
                raise RuntimeError("missing config.yaml")
            # OpenCOOD configs embed numpy objects that safe_load rejects (OpenCOOD's own
            # loader handles them); we only need the validate_dir line, so grep for it.
            with open(cfg) as f:
                text = f.read()
            m = re.search(r"^validate_dir:\s*['\"]?([^'\"\n]+?)['\"]?\s*$", text, re.M)
            if not m:
                raise RuntimeError("no validate_dir line in config.yaml")
            vdir = m.group(1)
            if not os.path.isdir(os.path.expanduser(vdir)):
                raise RuntimeError("validate_dir does not exist: %r (edit config.yaml)" % vdir)
            pths = [f for f in os.listdir(ckpt_dir) if f.endswith((".pth", ".pt"))]
            if not pths:
                raise RuntimeError("no .pth weights file")
            state = torch.load(os.path.join(ckpt_dir, sorted(pths)[-1]), map_location="cpu")
            n = len(state.get("model_state_dict", state)) if isinstance(state, dict) else 0
            return "%s loads (%d tensors), validate_dir ok" % (sorted(pths)[-1], n)
        return load

    if not os.path.isdir(root):
        check("checkpoint root", lambda: (_ for _ in ()).throw(RuntimeError("not a directory: %s" % root)))
        return
    subdirs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    if not subdirs:
        check("checkpoint root", lambda: (_ for _ in ()).throw(RuntimeError("no checkpoint folders in %s" % root)))
        return
    for d in subdirs:
        check("checkpoint: %s" % d, one(os.path.join(root, d)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["env", "dataset", "checkpoints", "all"], required=True)
    ap.add_argument("--dataset-root", help="OPV2V root (contains test/)")
    ap.add_argument("--checkpoint-root", help="folder of checkpoint folders")
    args = ap.parse_args()

    if args.stage in ("env", "all"):
        stage_env()
    if args.stage in ("dataset", "all"):
        if not args.dataset_root:
            ap.error("--dataset-root required for dataset stage")
        stage_dataset(args.dataset_root)
    if args.stage in ("checkpoints", "all"):
        if not args.checkpoint_root:
            ap.error("--checkpoint-root required for checkpoints stage")
        stage_checkpoints(args.checkpoint_root)

    print("\n=== Phase 0 gate report (stage: %s) ===" % args.stage)
    for status, name, detail in _results:
        print("[%s] %-35s %s" % (status, name, detail))
    failed = sum(1 for s, _, _ in _results if s == FAIL)
    print("=== %d/%d passed ===" % (len(_results) - failed, len(_results)))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
