
import setuptools
from setuptools import setup


print(setuptools.find_packages(where='autoencoder-procgen'))


setup(name='gym_autoencoder',
      version='0.1.8',
      install_requires=['gym','procgen','torch>=1.7.0','opencv-python'],
      author="Dennis Loran",
      packages=setuptools.find_packages(where='.'),
      package_data={'autoencoder-procgen': ['models/*.pt']}
      )
