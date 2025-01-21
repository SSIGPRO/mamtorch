# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import glob

from setuptools import find_packages, setup

from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)

library_name = "mamtorch"
version = "1.6.3"


def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"
    use_cuda = os.getenv("USE_CUDA", "1") == "1"
    if debug_mode:
        print("Compiling in debug mode")

    use_cuda = use_cuda and torch.cuda.is_available() and CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension

    extra_link_args = ["-lcusparse"]
    extra_compile_args = {
        "cxx": [
            "-O3" if not debug_mode else "-O0",
            "-fdiagnostics-color=always",
        ],
        "nvcc": [
            "-O3" if not debug_mode else "-O0",
        ],
    }
    if debug_mode:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["nvcc"].append("-g")
        extra_link_args.extend(["-O0", "-g"])

    this_dir = os.path.dirname(os.path.curdir)

    # kernel v1
    extensions_dir = os.path.join(this_dir, library_name, "kernel/v1/csrc")
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))
    extensions_cuda_dir = os.path.join(extensions_dir, "cuda")
    cuda_sources = list(glob.glob(os.path.join(extensions_cuda_dir, "*.cu")))
    if use_cuda:
        sources += cuda_sources

    # # kernel v2
    # extensions_dir_v2 = os.path.join(this_dir, library_name, "kernel/v2/csrc")
    # sources_v2 = list(glob.glob(os.path.join(extensions_dir_v2, "*.cpp")))
    # extensions_cuda_dir_v2 = os.path.join(extensions_dir_v2, "cuda")
    # cuda_sources_v2 = list(glob.glob(os.path.join(extensions_cuda_dir_v2, "*.cu")))
    # if use_cuda:
    #     sources_v2 += cuda_sources_v2

    # # kernel v3
    # extensions_dir_v3 = os.path.join(this_dir, library_name, "kernel/v3/csrc")
    # sources_v3 = list(glob.glob(os.path.join(extensions_dir_v3, "*.cpp")))
    # extensions_cuda_dir_v3 = os.path.join(extensions_dir_v3, "cuda")
    # cuda_sources_v3 = list(glob.glob(os.path.join(extensions_cuda_dir_v3, "*.cu")))
    # if use_cuda:
    #     sources_v3 += cuda_sources_v3

    # kernel v4
    # extensions_dir_v4 = os.path.join(this_dir, library_name, "kernel/v4/csrc")
    # sources_v4 = list(glob.glob(os.path.join(extensions_dir_v4, "*.cpp")))
    # extensions_cuda_dir_v4 = os.path.join(extensions_dir_v4, "cuda")
    # cuda_sources_v4 = list(glob.glob(os.path.join(extensions_cuda_dir_v4, "*.cu")))
    # if use_cuda:
    #     sources_v4 += cuda_sources_v4

    # kernel v5
    extensions_dir_v5 = os.path.join(this_dir, library_name, "kernel/v5/csrc")
    sources_v5 = list(glob.glob(os.path.join(extensions_dir_v5, "*.cpp")))
    extensions_cuda_dir_v5 = os.path.join(extensions_dir_v5, "cuda")
    cuda_sources_v5 = list(glob.glob(os.path.join(extensions_cuda_dir_v5, "*.cu")))
    if use_cuda:
        sources_v5 += cuda_sources_v5

    # sparse kernel v1
    extensions_dir_sparsev1 = os.path.join(this_dir, library_name, "sparse/kernel/v1/csrc")
    sources_sparsev1 = list(glob.glob(os.path.join(extensions_dir_sparsev1, "*.cpp")))
    extensions_cuda_dir_sparsev1 = os.path.join(extensions_dir_sparsev1, "cuda")
    cuda_sources_sparsev1 = list(glob.glob(os.path.join(extensions_cuda_dir_sparsev1, "*.cu")))
    if use_cuda:
        sources_sparsev1 += cuda_sources_sparsev1

    ext_modules = [
        extension(
            f"{library_name}.kernel.v1._C",
            sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        ),
        # extension(
        #     f"{library_name}.kernel.v2._C",
        #     sources_v2,
        #     extra_compile_args=extra_compile_args,
        #     extra_link_args=extra_link_args,
        # ),
        # extension(
        #     f"{library_name}.kernel.v3._C",
        #     sources_v3,
        #     extra_compile_args=extra_compile_args,
        #     extra_link_args=extra_link_args,
        # ),
        # extension(
        #     f"{library_name}.kernel.v4._C",
        #     sources_v4,
        #     extra_compile_args=extra_compile_args,
        #     extra_link_args=extra_link_args,
        # ),
        extension(
            f"{library_name}.kernel.v5._C",
            sources_v5,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        ),
        extension(
            f"{library_name}.sparse.kernel.v1._C",
            sources_sparsev1,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]

    return ext_modules


setup(
    name=library_name,
    version=version,
    packages=find_packages(),
    ext_modules=get_extensions(),
    install_requires=["torch"],
    description="Multiply-And-Max/min (MAM) neuron torch library",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/SSIGPRO/mamtorch",
    cmdclass={"build_ext": BuildExtension},
)
