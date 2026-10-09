"""Package the Python side of the G1 real-robot controller."""

from setuptools import find_packages, setup

package_name = 'g1_control'

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
        ('share/' + package_name + '/config', ['config/control.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cjh',
    maintainer_email='cjh@foxmail.com',
    description='G1 RENet policy and remote inspection nodes',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'remote_inspector = g1_control.remote_inspector:main',
            'renet_policy = g1_control.policy_node:main',
        ],
    },
)
