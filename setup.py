#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue 2 May 2023

@author : minh.ngo
"""
from setuptools import setup

setup(name='smart',
    description='Models for scripting tennis matches',
    author="Minh NGO",
    author_email="ngoc-minh.ngo@insa-lyon.fr",
    version='1.0',
    packages=['smart.video'],
    package_dir={'smart.video': './Video'},
    install_requires=['keras', 'tensorflow', 'numpy', 'scipy',
                      'matplotlib', 'opencv-python', 'imageio']
    )
