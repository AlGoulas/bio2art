#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from setuptools import setup
from setuptools import find_packages

setup(name='bio2art',
      version='0.1',
      description='Convert biological to artifial neural networks',
      author='Alexandros Goulas',
      author_email='agoulas227@gmail.com',
      classifiers=[
          'Intended Audience :: Science/Research/Any',
          'Programming Language :: Python :: 3.7.3',
      ],
      packages=find_packages(),
      include_package_data=True,
      package_data={'': ['connectomes/*.npy', 'connectomes/*.lst']},
)