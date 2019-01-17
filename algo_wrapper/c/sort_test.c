//
// Created by baojian on 8/16/18.
//

#include "sort.h"


int main() {
    double x[] = {0.2, -0.3, -0.1, 0.4, 0.1, 0.0};
    int x_len = 6, *sorted_ind = malloc(sizeof(int) * 6);
    arg_sort_descend(x, sorted_ind, x_len);
    for (int i = 0; i < x_len; i++) {
        printf("x[%d]: %lf sorted_ind: %d sorted_val: %lf\n",
               i, x[i], sorted_ind[i], x[sorted_ind[i]]);
    }
    printf("----\n");
    arg_sort_ascend(x, sorted_ind, x_len);
    for (int i = 0; i < x_len; i++) {
        printf("x[%d]: %lf sorted_ind: %d sorted_val: %lf\n",
               i, x[i], sorted_ind[i], x[sorted_ind[i]]);
    }
    printf("----\n");
    arg_magnitude_sort_descend(x, sorted_ind, x_len);
    for (int i = 0; i < x_len; i++) {
        printf("x[%d]: %lf sorted_ind: %d sorted_val: %lf\n",
               i, x[i], sorted_ind[i], x[sorted_ind[i]]);
    }
    printf("----\n");
    arg_magnitude_sort_ascend(x, sorted_ind, x_len);
    for (int i = 0; i < x_len; i++) {
        printf("x[%d]: %lf sorted_ind: %d sorted_val: %lf\n",
               i, x[i], sorted_ind[i], x[sorted_ind[i]]);
    }
    free(sorted_ind);
    printf("----\n");
    int k = 3, sorted_set_k[3];
    arg_magnitude_sort_top_k(x, sorted_set_k, k, x_len);
    for (int i = 0; i < k; i++) {
        printf("x[%d]: %lf sorted_ind: %d sorted_val: %lf\n",
               i, x[i], sorted_set_k[i], x[sorted_set_k[i]]);
    }
    printf("----\n");
    double xx[] = {9, 3, 1, 3, 4, 3, 6};
    double yy[] = {4, 6, 9, 2, 1, 8, 7};
    int xx_len = 7, sorted_indices[7];
    lex_sort_descend(yy, xx, sorted_indices, xx_len);
    for (int i = 0; i < xx_len; i++) {
        printf("xx[%d]: %lf yy[%d]: %lf sorted_ind: %d\n",
               i, xx[i], i, yy[i], sorted_indices[i]);
    }
    printf("----\n");
    lex_sort_ascend(yy, xx, sorted_indices, xx_len);
    for (int i = 0; i < xx_len; i++) {
        printf("xx[%d]: %lf yy[%d]: %lf sorted_ind: %d\n",
               i, xx[i], i, yy[i], sorted_indices[i]);
    }
}