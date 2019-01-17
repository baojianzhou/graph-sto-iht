//
// Created by baojian on 8/15/18.
//
#include "sort.h"

typedef struct {
    double val;
    int index;
} data_pair;

typedef struct {
    double first;
    double second;
    int index;
} lex_pair;

static inline int __comp_descend(const void *a, const void *b) {
    if (((data_pair *) a)->val < ((data_pair *) b)->val) {
        return 1;
    } else {
        return -1;
    }
}

static inline int __comp_ascend(const void *a, const void *b) {
    if (((data_pair *) a)->val < ((data_pair *) b)->val) {
        return -1;
    } else {
        return 1;
    }
}

static inline int __comp_lex_descend(const void *a, const void *b) {
    if ((((lex_pair *) a)->first < ((lex_pair *) b)->first) ||
        ((((lex_pair *) a)->first == ((lex_pair *) b)->first) &&
         ((lex_pair *) a)->second < ((lex_pair *) b)->second)) {
        return 1;
    } else {
        return -1;
    }
}

static inline int __comp_lex_ascend(const void *a, const void *b) {
    if ((((lex_pair *) a)->first < ((lex_pair *) b)->first) ||
        ((((lex_pair *) a)->first == ((lex_pair *) b)->first) &&
         ((lex_pair *) a)->second < ((lex_pair *) b)->second)) {
        return -1;
    } else {
        return 1;
    }
}

bool arg_sort_descend(const double *x, int *sorted_indices, int x_len) {
    data_pair *w_pairs = malloc(sizeof(data_pair) * x_len);
    for (int i = 0; i < x_len; i++) {
        w_pairs[i].val = x[i];
        w_pairs[i].index = i;
    }
    qsort(w_pairs, (size_t) x_len, sizeof(data_pair), &__comp_descend);
    for (int i = 0; i < x_len; i++) {
        sorted_indices[i] = w_pairs[i].index;
    }
    free(w_pairs);
    return true;
}

bool arg_sort_ascend(const double *x, int *sorted_indices, int x_len) {
    data_pair *w_pairs = malloc(sizeof(data_pair) * x_len);
    for (int i = 0; i < x_len; i++) {
        w_pairs[i].val = x[i];
        w_pairs[i].index = i;
    }
    qsort(w_pairs, (size_t) x_len, sizeof(data_pair), &__comp_ascend);
    for (int i = 0; i < x_len; i++) {
        sorted_indices[i] = w_pairs[i].index;
    }
    free(w_pairs);
    return true;
}

bool arg_magnitude_sort_descend(const double *x,
                                int *sorted_indices, int x_len) {
    data_pair *w_pairs = malloc(sizeof(data_pair) * x_len);
    for (int i = 0; i < x_len; i++) {
        w_pairs[i].val = fabs(x[i]);
        w_pairs[i].index = i;
    }
    qsort(w_pairs, (size_t) x_len, sizeof(data_pair), &__comp_descend);
    for (int i = 0; i < x_len; i++) {
        sorted_indices[i] = w_pairs[i].index;
    }
    free(w_pairs);
    return true;
}

bool arg_magnitude_sort_ascend(const double *x,
                               int *sorted_indices, int x_len) {
    data_pair *w_pairs = malloc(sizeof(data_pair) * x_len);
    for (int i = 0; i < x_len; i++) {
        w_pairs[i].val = fabs(x[i]);
        w_pairs[i].index = i;
    }
    qsort(w_pairs, (size_t) x_len, sizeof(data_pair), &__comp_ascend);
    for (int i = 0; i < x_len; i++) {
        sorted_indices[i] = w_pairs[i].index;
    }
    free(w_pairs);
    return true;
}


bool arg_magnitude_sort_top_k(const double *x,
                              int *sorted_set, int k, int x_len) {
    if (k > x_len) {
        printf("Error: k should be <= x_len\n");
        exit(EXIT_FAILURE);
    }
    data_pair *w_tmp = malloc(sizeof(data_pair) * x_len);
    for (int i = 0; i < x_len; i++) {
        w_tmp[i].val = fabs(x[i]);
        w_tmp[i].index = i;
    }
    qsort(w_tmp, (size_t) x_len, sizeof(data_pair), &__comp_descend);
    for (int i = 0; i < k; i++) {
        sorted_set[i] = w_tmp[i].index;
    }
    free(w_tmp);
    return true;
}

bool lex_sort_descend(const double *y, const double *x, int *sorted_indices,
                      int x_len) {
    lex_pair *elements = malloc(sizeof(lex_pair) * x_len);
    for (int i = 0; i < x_len; i++) {
        elements[i].first = x[i];
        elements[i].second = y[i];
        elements[i].index = i;
    }
    qsort(elements, (size_t) x_len, sizeof(lex_pair), &__comp_lex_descend);
    for (int i = 0; i < x_len; i++) {
        sorted_indices[i] = elements[i].index;
    }
    free(elements);
    return true;
}

bool lex_sort_ascend(const double *y, const double *x,
                     int *sorted_indices, int x_len) {
    lex_pair *elements = malloc(sizeof(lex_pair) * x_len);
    for (int i = 0; i < x_len; i++) {
        elements[i].first = x[i];
        elements[i].second = y[i];
        elements[i].index = i;
    }
    qsort(elements, (size_t) x_len, sizeof(lex_pair), &__comp_lex_ascend);
    for (int i = 0; i < x_len; i++) {
        sorted_indices[i] = elements[i].index;
    }
    free(elements);
    return true;
}
