#!/usr/bin/env python3
# -*- coding: utf-8
'''

'''
# Standard imports.
from setuptools import setup, find_packages
import sys

# Custom imports.
from ray_tracer import __version__, __author__, __url__

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name="unmagnetised_plasma_ray_tracing",
    version=__version__,
    description="Microwave ray tracing code for unmagnetised cold plasmas",
    long_description=readme(),
    url=__url__,
    packages=["unmagnetised_plasma_ray_tracing"],
    author=__author__,
    author_email="thomas.wilson@ukaea.uk",
    python_requires=">=3.9",
    install_requires=[
        "matplotlib",
        "numpy >=1.26.4",
        "scipy >=1.13",
        "netCDF4"
    ]
)
