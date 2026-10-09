"""Package the G1 depth-policy preprocessing node."""

from setuptools import find_packages, setup

package_name = 'g1_depth'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        ('share/' + package_name, ['package.xml']),
        (
            'share/' + package_name + '/config',
            [
                'config/depth_policy.yaml',
                'config/realsense_d435i.yaml',
                'config/realsense_d435i_temporal_on.yaml',
            ],
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cjh',
    maintainer_email='cjh@foxmail.com',
    description='G1 D435i policy-depth preprocessing',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'depth_policy_preprocessor = '
            'g1_depth.depth_policy_preprocessor:main',
        ],
    },
)
