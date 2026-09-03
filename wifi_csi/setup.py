from glob import glob
from setuptools import setup

package_name = 'wifi_csi'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wicoms',
    maintainer_email='you@example.com',
    description='Publishes Nexmon WiFi CSI as ROS 2 topics.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'csi_publisher = wifi_csi.csi_publisher:main',
            'csi_monitor  = wifi_csi.csi_monitor:main',
        ],
    },
)
