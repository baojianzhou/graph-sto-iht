/**
 *
 * @author: ---
 * @date:   --/--/2018.
 * @email:  --@--
 * @version: 1.12.0
 ===========================================================================*/
#include <cblas.h>
#include "loss.h"
#include "algo_best_subset.h"


bool algo_online_best_subset_logit(best_subset *para, ReStat *stat) {
    clock_t start_time = clock();
    openblas_set_num_threads(1);
    int p = para->p;
    int num_tr = para->num_tr;
    double *wt;
    double *wt_bar;
    double *gt_bar;
    double *loss_grad;
    double *wt_tmp;
    double *x_tr = para->x_tr;
    double *y_tr = para->y_tr;
    int *best_subset = para->best_subset;
    int best_subset_len = para->best_subset_len;

    wt = malloc(sizeof(double) * (p + 1));
    wt_bar = malloc(sizeof(double) * (p + 1));
    gt_bar = malloc(sizeof(double) * (p + 1));
    loss_grad = malloc(sizeof(double) * (p + 2));
    wt_tmp = malloc(sizeof(double) * (p + 1));

    cblas_dcopy(p + 1, para->w0, 1, wt, 1);         // w0 --> x_hat
    cblas_dcopy(p + 1, para->w0, 1, wt_bar, 1);     // w0 --> x_bar
    cblas_dscal(p + 1, 0.0, gt_bar, 1);             // 0 --> gt_bar

    for (int tt = 0; tt < num_tr; tt++) {
        double *x_i = x_tr + tt * p;
        double *y_i = y_tr + tt;
        // 2. observe y_i = y_tr + tt and have some loss
        logistic_loss_grad(wt, x_i, y_i, loss_grad, para->l2_lambda, 1, p);
        stat->losses[tt] = loss_grad[0];
        cblas_dscal(p + 1, tt / (tt + 1.), gt_bar, 1);
        cblas_daxpy(p + 1, 1. / (tt + 1.), loss_grad + 1, 1, gt_bar, 1);
        // 3. update model
        cblas_dcopy(p + 1, gt_bar, 1, wt_tmp, 1); // gt_bar --> wt_tmp
        cblas_dscal(p + 1, -sqrt(tt + 1.) / para->gamma, wt_tmp, 1);
        cblas_dscal(p + 1, 0.0, wt, 1);
        for (int i = 0; i < best_subset_len; i++) {
            wt[best_subset[i]] = wt_tmp[best_subset[i]];
        }
        wt[p] = wt_tmp[p];
        cblas_dscal(p + 1, tt / (tt + 1.), wt_bar, 1);
        cblas_daxpy(p + 1, 1 / (tt + 1.), wt, 1, wt_bar, 1);
    }
    // save results
    for (int i = 0; i < p + 1; i++) {
        stat->wt[i] = wt[i];
        stat->wt_bar[i] = wt_bar[i];
    }
    stat->total_time = (double) (clock() - start_time) / CLOCKS_PER_SEC;
    free(loss_grad);
    free(wt);
    free(wt_bar);
    free(wt_tmp);
    free(gt_bar);
    return true;
}