from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = [
    'tensorflow==2.7.2',
    'tensorflow_probability',
    'matplotlib',
    'gym[all]',
    'scikit-video',
    'dm-tree']


setup(
    name='on_policy',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    description='On Policy Hierarchal RL Algorithms')
