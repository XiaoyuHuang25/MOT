#!/usr/bin/env python
# Copyright (c) Megvii, Inc. and its affiliates. All Rights Reserved

# NOTE:
# This setup script has been extended to trigger the compilation of the
# Murty C++ pybind11 module before building YOLOX custom CUDA/C++ ops.
# The Murty module is built via the external `build_murty.sh` script
# (driven by the current Python interpreter), so that it can be imported from Python afterwards.

import re
import setuptools
import sys
import os
import subprocess
from pathlib import Path

MURTY_BUILD_SCRIPT = Path(__file__).with_name("build_murty.sh")


TORCH_AVAILABLE = True
try:
    import torch
    from torch.utils import cpp_extension
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] Unable to import torch, pre-compiling ops will be disabled.")


def get_package_dir():
    pkg_dir = {
        "yolox.tools": "tools",
        "yolox.exp.default": "exps/default",
    }
    return pkg_dir


def get_install_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        reqs = [x.strip() for x in f.read().splitlines()]
    reqs = [x for x in reqs if not x.startswith("#")]
    return reqs


def get_yolox_version():
    with open("yolox/__init__.py", "r") as f:
        version = re.search(
            r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
            f.read(), re.MULTILINE
        ).group(1)
    return version


def get_long_description():
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
    return long_description


def get_ext_modules():
    """
    Collect extension modules to be built with setuptools.

    - On non-Windows platforms:
        * Ensure torch is available.
        * Build the Murty C++ module via the external bash script.
        * Build YOLOX's FastCOCOEvalOp extension.
    - On Windows: no precompiled ops are built.
    """
    ext_module = []
    if sys.platform != "win32":  # pre-compile ops on Linux / macOS
        assert TORCH_AVAILABLE, (
            "torch is required for pre-compiling ops, please install it first."
        )

        # 1) Build the Murty C++ module (CMake + pybind11) before YOLOX ops.
        try:
            build_murty_cpp()
        except Exception as e:
            # You can decide whether to raise here or just warn.
            # Raising will make installation fail if Murty cannot be built.
            print(f"[WARNING] Failed to build Murty C++ module: {e}")

        # 2) Build YOLOX's built-in FastCOCOEvalOp.
        from yolox.layers import FastCOCOEvalOp
        ext_module.append(FastCOCOEvalOp().build_op())

    return ext_module


def get_cmd_class():
    cmdclass = {}
    if TORCH_AVAILABLE:
        cmdclass["build_ext"] = cpp_extension.BuildExtension
    return cmdclass


def build_murty_cpp():
    """
    Build the Murty C++ pybind11 module via the existing bash script.

    This function:
      - Only runs on non-Windows platforms.
      - Uses the current Python interpreter to drive the build.
      - Skips the build if a shared library (*.so) already exists
        in the specified build directory.
    """
    if sys.platform == "win32":
        print("[INFO] Skipping Murty build on Windows.")
        return

    # Change this path to match your actual Murty CMake build directory.
    build_dir = Path("murty/build").resolve()
    build_dir.mkdir(parents=True, exist_ok=True)

    output_dir = Path("./").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # If a shared library already exists, skip rebuilding.
    so_list = list(output_dir.glob("*.so*"))
    if so_list:
        print(f"[INFO] Murty module already built in {output_dir}, skipping rebuild.")
        return

    if not MURTY_BUILD_SCRIPT.exists():
        raise FileNotFoundError(
            f"Murty build script not found: {MURTY_BUILD_SCRIPT}"
        )

    print(f"[INFO] Building Murty C++ module in {build_dir} ...")
    cmd = [
        "bash",
        str(MURTY_BUILD_SCRIPT),
        "-b",
        str(build_dir),
        "-o",
        str(output_dir),
        "-p",
        sys.executable,  # use the current Python interpreter
    ]
    subprocess.run(cmd, check=True)
    print("[INFO] Murty C++ module build finished.")

setuptools.setup(
    name="yolox",
    version=get_yolox_version(),
    author="megvii basedet team",
    url="https://github.com/Megvii-BaseDetection/YOLOX",
    package_dir=get_package_dir(),
    packages=setuptools.find_packages(exclude=("tests", "tools")) + list(get_package_dir().keys()),
    python_requires=">=3.6",
    install_requires=get_install_requirements(),
    setup_requires=["wheel"],  # avoid building error when pip is not updated
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    include_package_data=True,  # include files in MANIFEST.in
    ext_modules=get_ext_modules(),
    cmdclass=get_cmd_class(),
    classifiers=[
        "Programming Language :: Python :: 3", "Operating System :: OS Independent",
        "License :: OSI Approved :: Apache Software License",
    ],
    project_urls={
        "Documentation": "https://yolox.readthedocs.io",
        "Source": "https://github.com/Megvii-BaseDetection/YOLOX",
        "Tracker": "https://github.com/Megvii-BaseDetection/YOLOX/issues",
    },
)
