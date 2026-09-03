"""mircpipe - shared layer for the MIRC dataset pipeline.

Every stage and every analysis imports from here instead of re-implementing
bag reading, SE(3) maths, map registration, or report writing.

    from mircpipe import se3, bag, cache, mapref, report
    from mircpipe.config import load_config

Nothing in this package knows about a particular stage; stage logic lives in
stages/, analysis logic in analysis/, and both are driven by run_pipeline.py.
"""
__all__ = ["config", "se3", "bag", "cache", "mapref", "report"]
__version__ = "1.0"
