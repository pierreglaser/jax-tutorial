#!/usr/bin/env python
# -*- coding: utf-8 -*-
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

description = ("Some sanity-checking-scripts to ensure that jax is properly installed")

dist = setup(
    name="jax-utils",
    version="0.0.0dev0",
    description=description,
    author="Pierre Glaser",
    author_email="pierreglaser@msn.com",
    license="BSD 3-Clause License",
    packages=["jax_utils"],
    install_requires=["jax"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    python_requires=">=3.8",
)
