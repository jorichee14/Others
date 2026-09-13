import os
from glob import glob

from setuptools import find_packages, setup

package_name = "comms"

setup(
    name=package_name,
    version="1.0.0",
    # comms, comms.wifi, comms.ntp, comms.csi
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
        (
            os.path.join("share", package_name, "config"),
            glob("config/*.yaml"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="jorichee14",
    maintainer_email="jorichloq@gmail.com",
    description="Wi-Fi link, throughput, latency, NTP and CSI monitoring nodes.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # wifi
            "wifi_monitor_node = comms.wifi.wifi_monitor_node:main",
            "iperf_runner_node = comms.wifi.iperf_runner_node:main",
            "ping_monitor_node = comms.wifi.ping_monitor_node:main",
            "burst_monitor_node = comms.wifi.burst_monitor_node:main",
            # ntp
            "ntp_client_node = comms.ntp.ntp_client_node:main",
            "ntp_server_node = comms.ntp.ntp_server_node:main",
            # csi
            "csi_publisher = comms.csi.csi_publisher:main",
            "csi_monitor = comms.csi.csi_monitor:main",
            "csi_view = comms.csi.csi_view:main",
        ],
    },
)
