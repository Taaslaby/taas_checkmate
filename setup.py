#!/usr/bin/env python3
"""Setup script for taas CLI tools and interactive TUI setup wizard."""

import sys

# If setup.py is invoked directly without commands (python setup.py) or with setup/tui command, launch TUI wizard!
if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1] in ["setup", "tui"]):
    from taas.setup_tui import run_tui
    try:
        run_tui()
    except KeyboardInterrupt:
        print("\n\n[!] Setup cancelled by user.")
    sys.exit(0)

from setuptools import setup, find_packages

setup(
    name="taas-tools",
    version="0.1.0",
    description="CLI toolkit for taas-checkmate neural engine",
    license="GPL-3.0-only",
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
