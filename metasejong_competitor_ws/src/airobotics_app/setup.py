from setuptools import find_packages, setup

package_name = 'airobotics_app'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='aisl',
    maintainer_email='thyun@sejong.ac.kr',
    description='AI Robotics App for Metasejong Competition',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'airobotics_node = airobotics_node.airobotics_node:main'
        ],
    },
)
