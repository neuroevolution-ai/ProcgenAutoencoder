

import setuptools
from setuptools import setup

import os



setup(name='gym_autoencoder',
      version='0.1.10',
      install_requires=['gym','procgen','torch>=1.7.0','opencv-python',"torchinfo","PyYAML","matplotlib","torchvision"],
      author="Dennis Loran",
      packages=["gym_autoencoder","model_stubs","gym_autoencoder.heist.envs"],
      package_data={'gym_autoencoder.heist.envs': ['../../../models/*.pt']},
      )
