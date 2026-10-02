# -*- coding: utf-8 -*-
"""
``ros2opv2v`` — convert a rosbag2 multi-agent recording into the OPV2V directory
layout that OpenCOOD reads.

Entry points:

* ``scripts/inspect_bag.py``     — what is in the bag, and a config skeleton for it
* ``scripts/convert_rosbag.py``  — run the conversion
* ``scripts/validate_opv2v.py``  — check the result the way OpenCOOD will read it

See ``docs/ROS2OPV2V.md`` for the conventions this package commits to (pose
parameterisation, intensity encoding, pseudo ground truth, ground lift).
"""

from .config import ConfigError, ConverterConfig, load_config          # noqa: F401
from .convert import ConversionError, ConversionReport, convert        # noqa: F401

__all__ = ["ConfigError", "ConverterConfig", "load_config",
           "ConversionError", "ConversionReport", "convert"]
