cmake_minimum_required(VERSION 2.8)
include_directories(algo_wrapper)
include_directories(algo_wrapper/c)
include_directories(head_tail_wrapper/c)
project(online-opt)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()
set(CMAKE_C_FLAGS "-Wall -Wextra")
set(CMAKE_C_FLAGS_DEBUG "-g")
set(CMAKE_C_FLAGS_RELEASE "-O3")
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11 -O3 -fPIC -lpython2.7 -pthread")
set(ROOT /network/rit/lab/ceashpc/bz383376/opt/)
set(NUMPY env-python2.7.14/lib/python2.7/site-packages/numpy/core/include/)
set(OPENBLAS_INCLUDE_DIR "${ROOT}openblas-0.3.1/include")
set(OPENBLAS_INCLUDE "${ROOT}openblas-0.3.1/include")
set(OPENBLAS_LIB "${ROOT}openblas-0.3.1/lib/")
set(NUMPY_INCLUDE "${ROOT}${NUMPY}")
set(PYTHON_LIBRARY "${ROOT}python-2.7.14/lib/libpython2.7.so")
set(PYTHON_INCLUDE "${ROOT}python-2.7.14/include/python2.7")
include_directories(${OPENBLAS_INCLUDE_DIR})
find_package(PythonLibs 2.7 REQUIRED)
message(${PYTHON_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIRS})

set(ROOT algo_wrapper/c/)
set(SRC_PCST ${ROOT}fast_pcst.c ${ROOT}fast_pcst.h)
set(SRC_Head ${ROOT}head_tail_proj.c ${ROOT}head_tail_proj.h)
set(SRC_Sort ${ROOT}sort.c ${ROOT}sort.h)
set(SRC_Loss ${ROOT}loss.c ${ROOT}loss.h)
set(SRC_Algo ${ROOT}sparse_algorithms.c ${ROOT}sparse_algorithms.h)
set(SRC_IHT ${ROOT}algo_sto_iht.c ${ROOT}algo_sto_iht.h)
set(SRC_Graph_IHT ${ROOT}algo_graph_sto_iht.c ${ROOT}algo_graph_sto_iht.h)
set(SRC_BEST_SUBSET ${ROOT}algo_best_subset.c ${ROOT}algo_best_subset.h)
set(SRC_CG ${ROOT}algo_conjugate_grad.c ${ROOT}algo_conjugate_grad.h)


add_executable(test_fast_pcst ${ROOT}fast_pcst_test.c ${SRC_PCST})
target_link_libraries(test_fast_pcst -std=c11 -Wall -Wextra -O3 -lm)

add_executable(test_head_tail ${ROOT}head_tail_proj_test.c
        ${SRC_PCST} ${SRC_Head} ${SRC_Sort})
target_link_libraries(test_head_tail -std=c11 -Wall -Wextra -O3 -lm)

add_executable(test_sort ${SRC_Sort} ${ROOT}sort_test.c)
target_link_libraries(test_sort -std=c11 -Wall -Wextra -O3 -lm)

add_executable(test_loss ${ROOT}loss_test.c ${SRC_Loss})
target_link_libraries(test_loss -std=c11 -Wall -Wextra -O3
        -I${OPENBLAS_INCLUDE} -L${OPENBLAS_LIB} -lm -lopenblas -lpthread)

set(SRC_1 ${SRC_PCST} ${SRC_Head})
set(SRC_2 ${SRC_Algo} ${SRC_Sort} ${SRC_Loss})
set(SRC_3 ${ROOT}math_utils.c ${ROOT}math_utils.h)

add_executable(test_sparse_algorithm
        ${ROOT}sparse_algorithms_test.c ${SRC_1} ${SRC_2} ${SRC_3})
target_link_libraries(test_sparse_algorithm
        python2.7 -std=c11 -Wall -Wextra -O3
        -I${OPENBLAS_INCLUDE} -I${NUMPY_INCLUDE}
        -L${OPENBLAS_LIB} -lm -lpython2.7 -lopenblas -lpthread)

add_executable(main_wrapper ${ROOT}main_wrapper.c ${SRC_1} ${SRC_2} ${SRC_3}
        ${SRC_IHT} ${SRC_BEST_SUBSET} ${SRC_Graph_IHT})
target_link_libraries(main_wrapper python2.7 -std=c11 -Wall -Wextra -O3
        -I${OPENBLAS_INCLUDE} -I${NUMPY_INCLUDE}
        -L${OPENBLAS_LIB} -lm -lpython2.7 -lopenblas -lpthread)

add_library(sparse_module SHARED ${ROOT}main_wrapper.c
        ${SRC_1} ${SRC_2} ${SRC_IHT} ${SRC_BEST_SUBSET} ${SRC_Graph_IHT})
target_link_libraries(sparse_module python2.7 -std=c11 -Wall -Wextra -O3
        -I${OPENBLAS_INCLUDE} -I${NUMPY_INCLUDE}
        -L${OPENBLAS_LIB} -lm -lpython2.7 -lopenblas -lpthread)
