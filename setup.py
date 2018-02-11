from setuptools import setup
from distutils.core import setup, Extension
 
cuda_matmul = Extension('cuda_matmul', sources=["cuda_matmul.c"],
                            include_dirs=['.'],
                            libraries=['my_matmul'],
                            library_dirs=['.'],
                            runtime_library_dirs=["$ORIGIN"],
)
 
setup(name="cuda_matmul", version=1.0,
description="Dummy implementation of matrix multiplication for python using CUDA",
ext_modules=[cuda_matmul])
