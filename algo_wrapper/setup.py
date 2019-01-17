# -*- coding: utf-8 -*-
"""
Some documents will be added later.
"""
import os
import numpy
from os import path
from setuptools import setup
from distutils.core import Extension

here = path.abspath(path.dirname(__file__))


def get_c_sources(folder, include_headers=False):
    """Find all C/C++ source files in the `folder` directory."""
    allowed_extensions = [".c", ".C", ".cc", ".cpp", ".cxx", ".c++"]
    if include_headers:
        allowed_extensions.extend([".h", ".hpp"])
    sources = []
    for root, dirs, files in os.walk(folder):
        for name in files:
            ext = os.path.splitext(name)[1]
            if ext in allowed_extensions:
                sources.append(os.path.join(root, name))
    return sources


root_path = "/network/rit/lab/ceashpc/bz383376/opt/"
packages = root_path + "env-python2.7.14/lib/python2.7/site-packages/"
numpy_include = packages + "numpy/core/include/"
openblas_include = root_path + "openblas-0.3.1/include"
python_lib = root_path + "python-2.7.14/lib/"
openblas_lib = root_path + "openblas-0.3.1/lib/libopenblas.so"

# calling the setup function
setup(
    # sparse_learning package.
    name='sparse_module.so',
    # current version is 0.2.1
    version='0.2.2',
    # this is a wrapper of head and tail projection.
    description='A wrapper for sparse learning algorithms.',
    # a long description should be here.
    long_description='This package collects sparse learning algorithms.',
    # url of github projection.
    url='????',
    # number of authors.
    author='???? ????',
    # my email.
    author_email='????@????',
    include_dirs=[numpy.get_include()],
    license='MIT',
    packages=['sparse_learning'],
    classifiers=("Programming Language :: Python :: 2",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: POSIX :: Linux",),
    # specify requirements of your package here
    # will add openblas in later version.
    install_requires=['numpy'],
    headers=['../fast_pcst.h',
             '../head_tail_proj.h',
             '../loss.h',
             '../sort.h',
             '../sparse_algorithms.h'],
    # define the extension module
    ext_modules=[
        Extension('proj_module',
                  sources=['../fast_pcst.c',
                           '../head_tail_proj.c',
                           '../loss.c',
                           '../sort.c',
                           '../sparse_algorithms.c',
                           '../main_wrapper.c'],
                  language="c",
                  extra_compile_args=['-std=c11', '-shared', '-Wall', '-fPIC',
                                      '-O3', '-lpython2.7'],
                  include_dirs=[numpy.get_include(),
                                openblas_include,
                                numpy_include],
                  library_dirs=[openblas_lib,
                                python_lib]
                  )],
    keywords='sparse learning, structure sparsity, head/tail projection')
