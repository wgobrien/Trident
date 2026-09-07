from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.install import install


PIP_REQUIREMENTS = [
    "pyro-ppl>=1.9.1",
    "numpy>=1.21",
    "anndata",
    "h5py<3.15; python_version < '3.11'",
    "pandas",
    "seaborn",
    "graphviz",
    "requests",
    "fastcluster",
    "scipy",
    "polars",
    "pysam",
    "scikit-learn",
    "matplotlib-venn",
    "taxutils",
    "mappy",
]

CONDA_TOOLS = ["kraken2", "megahit", "minimap2"]

ROOT = Path(__file__).resolve().parent
BUILD_DATA_DIR = ROOT / "trident" / "tools" / "build_data"


def _build_cpp_binaries() -> None:
    make = shutil.which("make")
    if make is None:
        raise RuntimeError("`make` is required to compile trident C++ binaries")
    subprocess.check_call([make, "-C", str(BUILD_DATA_DIR), "all"])


def _install_conda_tools() -> None:
    if os.environ.get("TRIDENT_SKIP_CONDA_TOOLS", "").lower() in {"1", "true", "yes"}:
        print("Skipping conda tool installation (TRIDENT_SKIP_CONDA_TOOLS set).")
        return

    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda:
        raise RuntimeError(
            "Conda is required to install external tools "
            "(kraken2, megahit, minimap2). "
            "Install conda or set TRIDENT_SKIP_CONDA_TOOLS=1 to skip."
        )

    cmd = [
        conda,
        "install",
        "-y",
        "-c",
        "conda-forge",
        "-c",
        "bioconda",
        *CONDA_TOOLS,
    ]
    subprocess.check_call(cmd)


class BuildPyCommand(build_py):
    def run(self) -> None:
        _build_cpp_binaries()
        super().run()


class InstallCommand(install):
    def run(self) -> None:
        _build_cpp_binaries()
        _install_conda_tools()
        super().run()


class DevelopCommand(develop):
    def run(self) -> None:
        _build_cpp_binaries()
        _install_conda_tools()
        super().run()


setup(
    name="trident",
    version="1.0.0",
    description="A probabilistic program for taxonomic profiling in metagenomic reads.",
    author="William O'Brien",
    author_email="wob@cs.ucla.edu",
    license="MIT",
    url="https://github.com/wgobrien/trident",
    packages=find_packages(include=["trident*"]),
    include_package_data=True,
    package_data={"trident": ["tools/build_data/build_counts", "tools/build_data/coverage"]},
    python_requires=">=3.10",
    install_requires=PIP_REQUIREMENTS,
    entry_points={"console_scripts": ["trident=trident.cli:main"]},
    cmdclass={
        "build_py": BuildPyCommand,
        "install": InstallCommand,
        "develop": DevelopCommand,
    },
)
