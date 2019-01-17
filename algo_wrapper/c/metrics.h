//
// Created by baojian on 8/13/18.
//

#ifndef FAST_PCST_METRICS_H
#define FAST_PCST_METRICS_H

#include <stdlib.h>
#include <stdbool.h>

typedef enum {
    None, Micro, Macro, Samples, Weighted
} MetricType;

typedef enum {
    Binary, Multiclass, MultiIndicator, Continuous
} TargetType;

void _average_binary_score(double binary_metric,
                           double *y_true,
                           double *y_score,
                           MetricType average,
                           double *sample_weight);

void roc_curve(const double *y_true, const double *y_score,
               double posi_label, double *sample_weight,
               bool drop_intermediate, int n);

#endif //FAST_PCST_METRICS_H
