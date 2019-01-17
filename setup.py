# -*- coding: utf-8 -*-
"""
command:    python setup.py build_ext --inplace
"""
import os
import numpy
from setuptools import setup
from distutils.core import Extension

here = os.path.abspath(os.path.dirname(__file__))

src_files = ['algo_wrapper/c/main_wrapper.c',
             'algo_wrapper/c/head_tail_proj.c',
             'algo_wrapper/c/fast_pcst.c',
             'algo_wrapper/c/sort.c']
compile_args = ['-shared', '-Wall', '-g', '-fPIC',
                '-std=c11', '-lpython2.7', '-lm']
# calling the setup function
setup(
    # sparse_learning package.
    name='sparse_module',
    # current version is 0.0.1
    version='0.0.1',
    # this is a wrapper of head and tail projection.
    description='----',
    long_description='----',
    url='----',
    author='----',
    author_email='----',
    include_dirs=[numpy.get_include()],
    license='MIT',
    packages=['sparse_learning'],
    classifiers=("Programming Language :: Python :: 2",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: POSIX :: Linux",),
    # specify requirements of your package here
    install_requires=['numpy'],
    headers=['algo_wrapper/c/head_tail_proj.h',
             'algo_wrapper/c/fast_pcst.h',
             'algo_wrapper/c/sort.h'],
    # define the extension module
    ext_modules=[Extension('sparse_module',
                           sources=src_files,
                           language="C",
                           extra_compile_args=compile_args,
                           include_dirs=[numpy.get_include()])],
    keywords='sparse learning, structure sparsity, head/tail projection')
