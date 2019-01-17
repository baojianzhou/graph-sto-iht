#!/bin/bash
ROOT_PATH=/network/rit/lab/ceashpc/bz383376/opt/
PYTHON_PATH=${ROOT_PATH}python-2.7.14/include/python2.7
NUMPY_PATH=${ROOT_PATH}env-python2.7.14/lib/python2.7/site-packages/numpy/core/include/
PYTHON_LIB=${ROOT_PATH}python-2.7.14/lib/
FLAGS="-g -shared  -Wall -fPIC -std=c11 -O3 "
SRC_1="c/fast_pcst.c c/fast_pcst.h c/head_tail_proj.c c/head_tail_proj.h "
SRC="c/main_wrapper.c ${SRC_1} c/sort.c c/sort.h "
INCLUDE="-I${PYTHON_PATH} -I${NUMPY_PATH}"
LIB="-L${PYTHON_LIB} "
gcc ${FLAGS} ${INCLUDE} ${LIB} ${SRC} -o sparse_module.so -lm
