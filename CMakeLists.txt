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
set(NUMPY_INCLUDE "${ROOT}${NUMPY}")
set(PYTHON_LIBRARY "${ROOT}python-2.7.14/lib/libpython2.7.so")
set(PYTHON_INCLUDE "${ROOT}python-2.7.14/include/python2.7")
find_package(PythonLibs 2.7 REQUIRED)
message(${PYTHON_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIRS})

set(ROOT algo_wrapper/c/)
set(SRC_PCST ${ROOT}fast_pcst.c ${ROOT}fast_pcst.h)
set(SRC_Head ${ROOT}head_tail_proj.c ${ROOT}head_tail_proj.h)
set(SRC_Sort ${ROOT}sort.c ${ROOT}sort.h)

add_executable(test_fast_pcst ${ROOT}fast_pcst_test.c ${SRC_PCST})
target_link_libraries(test_fast_pcst -std=c11 -Wall -Wextra -O3 -lm)

add_executable(test_head_tail ${ROOT}head_tail_proj_test.c
        ${SRC_PCST} ${SRC_Head} ${SRC_Sort})
target_link_libraries(test_head_tail -std=c11 -Wall -Wextra -O3 -lm)

add_executable(test_sort ${SRC_Sort} ${ROOT}sort_test.c)
target_link_libraries(test_sort -std=c11 -Wall -Wextra -O3 -lm)


set(SRC_1 ${SRC_PCST} ${SRC_Head})

add_executable(main_wrapper ${ROOT}main_wrapper.c ${SRC_1} ${SRC_Sort})
target_link_libraries(main_wrapper python2.7 -std=c11 -Wall -Wextra -O3
        -I${NUMPY_INCLUDE} -lm -lpython2.7 -lpthread)

add_library(sparse_module SHARED ${ROOT}main_wrapper.c ${SRC_1} ${SRC_2})
target_link_libraries(sparse_module python2.7 -std=c11 -Wall -Wextra -O3
        -I${NUMPY_INCLUDE} -lm -lpython2.7 -lpthread)
