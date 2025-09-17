#!/usr/bin/env python3
"""
Buhera Framework Validation Package

A comprehensive Python package demonstrating and validating the core principles
of the Buhera VPOS quantum computing framework through practical implementations.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="buhera-framework-validation",
    version="1.0.0",
    author="Buhera Research Team",
    description="Validation package for revolutionary consciousness-substrate computing framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Operating Systems",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "networkx>=2.8.0",
        "scipy>=1.9.0",
        "pandas>=1.5.0",
        "pytest>=7.0.0",
        "tqdm>=4.64.0",
    ],
    extras_require={
        "dev": [
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
        "visualization": [
            "plotly>=5.10.0",
            "seaborn>=0.11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "buhera-demo=buhera_validation.cli:main",
            "buhera-benchmark=buhera_validation.benchmarks:main",
        ],
    },
)
