#!/bin/bash
ROOT_PATH=/network/rit/lab/ceashpc/bz383376/opt/
PYTHON_PATH=${ROOT_PATH}python-2.7.14/include/python2.7
NUMPY_PATH=${ROOT_PATH}env-python2.7.14/lib/python2.7/site-packages/numpy/core/include/
OPENBLAS_PATH=${ROOT_PATH}openblas-0.3.1/include/
OPENBLAS_LIB=${ROOT_PATH}openblas-0.3.1/lib/
PYTHON_LIB=${ROOT_PATH}python-2.7.14/lib/
FLAGS="-g -shared  -Wall -fPIC -std=c11 -O3 "
SRC_1="c/fast_pcst.c c/fast_pcst.h "
SRC_2="c/head_tail_proj.c c/head_tail_proj.h "
SRC_3="c/sparse_algorithms.c c/sparse_algorithms.h "
SRC_4="c/loss.c c/loss.h c/sort.c c/sort.h "
Math="c/math_utils.c c/math_utils.h "
IHT="c/algo_sto_iht.c c/algo_sto_iht.h "
GraphIHT="c/algo_graph_sto_iht.c c/algo_graph_sto_iht.h "
BestSubset="c/algo_best_subset.c c/algo_best_subset.h "
SRC="c/main_wrapper.c ${SRC_1}${SRC_2}${SRC_3}${SRC_4}${Math}${IHT}${GraphIHT}${BestSubset}"
INCLUDE="-I${PYTHON_PATH} -I${NUMPY_PATH} -I${OPENBLAS_PATH}"
LIB="-L${OPENBLAS_LIB} -L${PYTHON_LIB} "
gcc ${FLAGS} ${INCLUDE} ${LIB} ${SRC} -o sparse_module.so -lopenblas -lm
