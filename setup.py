from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = [
    'tensorflow==2.1',
    'tensorflow_probability',
    'gym[all]',
    'dm-control',
    'dm-tree']


setup(
    name='itch',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    description='Information Theoretic Control Hierarchies')
