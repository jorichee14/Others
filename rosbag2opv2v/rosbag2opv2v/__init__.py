"""Convert a rosbag2 (MCAP) multi-agent recording into an OPV2V dataset for OpenCOOD."""

__version__ = "0.1.0"

__all__ = ["Config", "Converter", "verify"]


def __getattr__(name):  # keep numpy/mcap imports lazy for `--help`
    if name == "Config":
        from .config import Config
        return Config
    if name == "Converter":
        from .convert import Converter
        return Converter
    if name == "verify":
        from .verify import verify
        return verify
    raise AttributeError(name)
