#!/usr/bin/env python
# -*- coding: utf-8 -*-

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="seqcond",
    version="0.1.0",
    author="Maixent Chenebaux",
    author_email="max.chbx@gmail.com",
    description="A TensorFlow library for sequence conditioning models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maixentchenebaux/seqcond",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.0",
        "numpy>=1.19.0",
    ],
    license="MIT",
    keywords="tensorflow deep-learning sequence-conditioning nlp",
)