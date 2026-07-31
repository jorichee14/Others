import os
from glob import glob

from setuptools import find_packages, setup

package_name = "wifi_monitor"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob("launch/*.launch.py"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="jorichee14",
    maintainer_email="jorichloq@gmail.com",
    description="ROS 2 wireless link monitor (RSSI, SNR, bit rate, traffic).",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "wifi_monitor_node = wifi_monitor.wifi_monitor_node:main",
            "iperf_runner_node = wifi_monitor.iperf_runner_node:main",
        ],
    },
)
