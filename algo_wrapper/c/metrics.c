//
// Created by baojian on 8/13/18.
//

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "sort.h"
#include "metrics.h"


/**
 * x :
    y : array, shape = [n]
        y coordinates.
    reorder : boolean, optional (default=False)
        If True, assume that the curve is ascending in the case of ties, as for
        an ROC curve. If the curve is non-ascending, the result will be wrong.
 * @param x_coordinates array, shape = [n] x coordinates.
 * @param y_coordinates array, shape = [n] x coordinates.
 * @param reorder
 * @return
 */
double auc(double *x_coordinates,
           double *y_coordinates,
           bool reorder, int x_len) {
    if (x_len < 2) {
        printf("error: At least 2 points are needed "
               "to compute AUC, but x.shape = %d", x_len);
    }

    return 0.0;
}

void _average_binary_score(double binary_metric,
                           double *y_true,
                           double *y_score,
                           MetricType average,
                           double *sample_weight){
    printf("test");
}

void binary_clf_curve(const double *y_true, const double *y_score,
                      double posi_label, double *sample_weight, int y_len) {
    double y_true_[y_len], y_score_[y_len], weight_[y_len];
    int desc_score_indices[y_len];
    // to make sure posi_label is 1 or -1
    if (posi_label != 1 && posi_label != -1) {
        //error
        return;
    }
    // to make y_true a boolean vector
    for (int i = 0; i < y_len; i++) {
        if (y_true[i] == posi_label) {
            y_true_[i] = true;
        } else {
            y_true_[i] = false;
        }
    }
    arg_sort_descend(y_score, desc_score_indices, y_len);
    for (int i = 0; i < y_len; i++) {
        int ind = desc_score_indices[i];
        y_score_[ind] = y_score[ind];
        y_true_[ind] = y_true[ind];
    }
    if (sample_weight != NULL) {
        for (int i = 0; i < y_len; i++) {
            weight_[i] = sample_weight[i];
        }
    } else {
        for (int i = 0; i < y_len; i++) {
            weight_[i] = 1.;
        }
    }
    int threshold_idxs[y_len], threshold_idxs_len = 0;
    threshold_idxs[threshold_idxs_len++] = 0;
    for (int i = 1; i < y_len; i++) {
        if (y_score_[i] != threshold_idxs[threshold_idxs_len - 1]) {
            threshold_idxs[threshold_idxs_len++] = i;
        }
    }
    double tps[threshold_idxs_len];
    int index1 = threshold_idxs[0], index2 = 0;
    tps[0] = 0.0;
    for (int i = 0; i < y_len; i++) {
        if (i != index1) {
            tps[index2] += y_true_[threshold_idxs[i]] * weight_[i];
        } else {
            index2++;
            index1 = threshold_idxs[index2];
            tps[index2] = tps[index2 - 1];
        }
    }
}

void roc_curve(const double *y_true, const double *y_score,
               double posi_label, double *sample_weight,
               bool drop_intermediate, int y_len) {
    double tmp_y_true[y_len], tmp_y_score[y_len];
    int desc_score_indices[y_len];
    // to make sure posi_label is 1 or -1
    if (posi_label != 1 && posi_label != -1) {
        //error
        return;
    }
    // to make y_true a boolean vector
    for (int i = 0; i < y_len; i++) {
        if (y_true[i] == posi_label) {
            tmp_y_true[i] = true;
        } else {
            tmp_y_true[i] = false;
        }
    }
    arg_sort_descend(y_score, desc_score_indices, y_len);
    for (int i = 0; i < y_len; i++) {
        int ind = desc_score_indices[i];
        tmp_y_true[ind] = y_true[ind];
        tmp_y_score[ind] = y_score[ind];
    }
    double fpr[y_len], tpr[y_len];
    auc(fpr, tpr, true, y_len);
}

void binary_roc_auc_score(
        const double *y_true,
        const double *y_score,
        const double *sample_weight) {

}

//Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)
//    from prediction scores
void roc_auc_score(const double *y_true, const double *y_score,
                   const double *sample_weight, MetricType average) {

}

