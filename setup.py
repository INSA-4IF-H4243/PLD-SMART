#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue 2 May 2023

@author : INSA Hexanome H4243 - 4IF
"""
from setuptools import setup

setup(name='smart',
    description='Models for scripting tennis matches',
    author="INSA Hexanome H4243 - 4IF",
    version='1.0',
    packages=['smart.video', 'smart.processor', 'smart.model'],
    package_dir={'smart.video': './video',
                 'smart.processor': './processor',
                 'smart.model': './model'},
    install_requires=['tensorflow', 'numpy', 'scipy',
                      'matplotlib', 'opencv-python',
                      'pynput', 'cvzone', 'mediapipe',
                      'scikit-learn', 'ffmpeg-python']
    )
