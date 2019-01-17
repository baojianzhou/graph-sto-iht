#!/bin/bash
sort_src="c/sort.h c/sort.c"
fast_pcst_src="c/fast_pcst.h c/fast_pcst.c"
head_tail_src="c/head_tail_proj.h c/head_tail_proj.c"
all_src="${sort_src} ${fast_pcst_src} ${head_tail_src}"
gcc -g -Wall -std=c11 -O3 ${sort_src} c/sort_test.c -o main_sort
gcc -g -Wall -std=c11 -O3 ${fast_pcst_src} ./c/fast_pcst_test.c -o main_fast_pcst
gcc -g -Wall -std=c11 -O3 ${all_src}  c/head_tail_proj_test.c -o main_head_tail_proj -lm