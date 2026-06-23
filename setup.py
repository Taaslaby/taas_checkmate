#!/usr/bin/env python3
"""Setup script for taas CLI tools.

This installs the `taas` command-line toolkit alongside the Rust module.

Usage:
    pip install -e .              # editable install
    taas --help                    # use the CLI
"""

from setuptools import setup, find_packages

setup(
    name="taas-tools",
    version="0.1.0",
    description="CLI toolkit for taas-checkmate neural engine",
    packages=find_packages(include=["taas"]),
    python_requires=">=3.10",
    install_requires=[
        "click>=8.0",
        "rich>=10.0",
        "numpy>=1.24",
    ],
    extras_require={
        "viz": ["matplotlib>=3.5"],
        "dashboard": ["streamlit>=1.20", "plotly>=5.10"],
        "notebooks": ["jupyter>=1.0", "nbformat>=5.0"],
        "eeg": ["scipy>=1.8"],
        "all": [
            "matplotlib>=3.5",
            "streamlit>=1.20",
            "plotly>=5.10",
            "jupyter>=1.0",
            "nbformat>=5.0",
            "scipy>=1.8",
        ],
    },
    entry_points={
        "console_scripts": [
            "taas=taas.cli:cli",
        ],
    },
)
