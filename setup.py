from setuptools import setup, find_packages

setup(
    name='ur_mvo',
    version='0.1.0',
    description='Package for visual localization and mapping for underwater scenes',
    author='Jaafar Mahmoud',
    author_email='jaafar.a.mahmoud1@gmail.com',
    packages=find_packages(where='./'),
    package_dir={'ur_mvo': '.'},
    entry_points={
        'console_scripts': [
            'vo_node = ur_mvo.vo:main'
        ],
    },
)
