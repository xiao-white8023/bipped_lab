from setuptools import find_packages, setup

package_name = 'lidar'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (
            'share/' + package_name + '/config',
            [
                'config/phase1_baseline.yaml',
                'config/phase2_rolling.yaml',
                'config/phase3_persistent.yaml',
            ],
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cjh',
    maintainer_email='cjh@foxmail.com',
    description='G1 MID-360 LiDAR policy-observation preprocessing',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'baseline_elevation_analyzer = '
            'lidar.baseline_elevation_analyzer:main',
            'cloud_preprocessor = lidar.cloud_preprocessor:main',
            'lidar_point = lidar.lidar_point:main',
            'persistent_elevation_map = '
            'lidar.persistent_elevation_map:main',
            'rolling_elevation_map = lidar.rolling_elevation_map:main',
            'static_elevation_map = lidar.static_elevation_map:main',
        ],
    },
)
