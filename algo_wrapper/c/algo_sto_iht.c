#include <cblas.h>
#include <time.h>
#include "loss.h"
#include "sort.h"
#include "math_utils.h"
#include "algo_sto_iht.h"

ReStoIHTStat *make_re_sto_iht_stat(int p, int b, int max_epochs) {
    ReStoIHTStat *re_stat = malloc(sizeof(ReStoIHTStat));
    re_stat->epoch_losses = malloc(sizeof(double) * max_epochs);
    re_stat->iter_losses = malloc(sizeof(double) * b * max_epochs);
    re_stat->x_epoch_errors = malloc(sizeof(double) * max_epochs);
    re_stat->x_iter_errors = malloc(sizeof(double) * b * max_epochs);
    re_stat->x_bar = malloc(sizeof(double) * (p + 1));
    re_stat->x_hat = malloc(sizeof(double) * (p + 1));
    re_stat->num_epochs = 0;
    re_stat->run_time = 0.0;
    return re_stat;
}

bool free_re_sto_iht_stat(ReStoIHTStat *re_stat) {
    free(re_stat->x_hat);
    free(re_stat->x_bar);
    free(re_stat->x_iter_errors);
    free(re_stat->x_epoch_errors);
    free(re_stat->iter_losses);
    free(re_stat->epoch_losses);
    free(re_stat);
    return true;
}

void hard_thresholding_operator(double *x, int *sorted_ind, int s, int p) {
    arg_magnitude_sort_descend(x, sorted_ind, p);
    for (int i = s; i < p; i++) {
        x[sorted_ind[i]] = 0.0;
    }
}

bool algo_sto_iht_logit(StoIHTPara *para, ReStoIHTStat *stat) {

    clock_t start_total = clock();
    openblas_set_num_threads(1);
    int p = para->p, s = para->s, num_tr = para->n, *sorted_ind;
    double *wt, *wt_bar, *loss_grad, *wt_tmp, eta = para->lambda_l2,
            gamma = para->lr;
    double *x_tr = para->x_tr, *y_tr = para->y_tr;

    wt = malloc(sizeof(double) * (p + 1));
    wt_bar = malloc(sizeof(double) * (p + 1));
    loss_grad = malloc(sizeof(double) * (p + 2));
    wt_tmp = malloc(sizeof(double) * (p + 1));
    sorted_ind = malloc(sizeof(int) * p);

    cblas_dcopy(p + 1, para->w0, 1, wt, 1);      // w0 --> x_hat
    cblas_dcopy(p + 1, para->w0, 1, wt_bar, 1);  // w0 --> x_bar

    for (int tt = 0; tt < num_tr; tt++) {
        // 1.   unlabeled example x_i = x_tr + tt*p arrives and
        //      make prediction based on existing costs x_hat, x_bar
        double *x_i = x_tr + tt * p, *y_i = y_tr + tt;
        // 2. observe y_i = y_tr + tt and have some loss
        logistic_loss_grad(wt, x_i, y_i, loss_grad, eta, 1, p);
        stat->epoch_losses[tt] = loss_grad[0];
        // 3. update x_hat= x_hat - lr/sqrt(tt) * grad(f_xi, x_hat)
        cblas_dcopy(p + 1, wt, 1, wt_tmp, 1); // x_hat --> wt_tmp
        cblas_daxpy(p + 1, -gamma / sqrt(tt + 1.), loss_grad + 1, 1, wt_tmp,
                    1);
        // 4. projection step: select largest k entries.
        arg_magnitude_sort_descend(wt_tmp, sorted_ind, p);
        cblas_dscal(p + 1, 0.0, wt, 1);
        for (int i = 0; i < s; i++) {
            wt[sorted_ind[i]] = wt_tmp[sorted_ind[i]];
        }
        // 5. intercept is not a feature. keep the intercept in entry p.
        wt[p] = wt_tmp[p];
        cblas_dscal(p + 1, tt / (tt + 1.), wt_bar, 1);
        cblas_daxpy(p + 1, 1 / (tt + 1.), wt, 1, wt_bar, 1);
    }
    for (int i = 0; i < p + 1; i++) {
        stat->x_hat[i] = wt[i];
        stat->x_bar[i] = wt_bar[i];
    }
    stat->run_time = (double) (clock() - start_total) / CLOCKS_PER_SEC;
    free(wt), free(wt_bar), free(loss_grad), free(wt_tmp), free(sorted_ind);
    return true;
}


bool algo_sto_iht_least_square(StoIHTPara *para, ReStoIHTStat *stat) {
    clock_t start_time = clock();
    openblas_set_num_threads(1);
    srand((unsigned int) time(0));

    int n = para->n;
    int s = para->s;
    int p = para->p;
    int b = para->b;

    double lr = para->lr;
    double *x_tr = para->x_tr;
    double *y_tr = para->y_tr;
    int verbose = para->verbose;
    double *x_star = para->x_star;
    double tol_algo = para->tol_algo;
    int max_epochs = para->max_epochs;
    double *prob_arr = para->prob_arr;
    bool with_replace = para->with_replace;

    /// if the block size is larger than n,
    /// just use a single block (batch)
    b = n < b ? n : b;
    int num_blocks = n / b;

    int *sorted_ind = malloc(sizeof(int) * p);
    /// random permutation.
    int *r_perm = malloc(sizeof(int) * num_blocks);
    double *tmp_xty = malloc(sizeof(double) * n);
    double *x_hat = malloc(sizeof(double) * (p));
    double *x_bar = malloc(sizeof(double) * (p));
    double *x_hat_tmp = malloc(sizeof(double) * (p));
    /// w0 --> x_hat and w0 --> x_bar
    cblas_dcopy(p, para->w0, 1, x_hat, 1);
    cblas_dcopy(p, para->w0, 1, x_bar, 1);
    for (int i = 0; i < num_blocks; i++) {
        r_perm[i] = i;
    }
    // each epochs
    int tt = 0; // a clock t: to keep track the current iterations.
    for (int epoch_i = 0; epoch_i < max_epochs; epoch_i++) {
        stat->num_epochs += 1;
        rand_shuffle(r_perm, num_blocks);
        /// current loss
        cblas_dcopy(n, y_tr, 1, tmp_xty, 1);
        cblas_dcopy(p, x_hat, 1, x_hat_tmp, 1);
        cblas_dgemv(CblasRowMajor, CblasNoTrans,
                    n, p, 1., x_tr, p, x_hat, 1, -1.0, tmp_xty, 1);
        double rooted_se = norm_l2(tmp_xty, n);
        stat->epoch_losses[epoch_i] = rooted_se * rooted_se;
        /// we have a true signal here. we compute the error of x ||x-x_hat||.
        if (x_star != NULL) {
            cblas_daxpy(p, -1., x_star, 1, x_hat_tmp, 1);
            stat->x_epoch_errors[epoch_i] = norm_l2(x_hat_tmp, p);
        }

        // inner epoch
        for (int in_ii = 0; in_ii < num_blocks; in_ii++) {
            /// random choose a block.
            int block_ii = (int) (with_replace ? random() % (num_blocks)
                                               : r_perm[in_ii]);
            cblas_dcopy(b, y_tr + block_ii * b, 1, tmp_xty, 1);
            /// calculate the residual: tmp_xty = A_S x_hat - y_S
            cblas_dgemv(CblasRowMajor, CblasNoTrans,
                        b, p, 1., x_tr + block_ii * b * p, p, x_hat, 1, -1.0,
                        tmp_xty, 1);
            double c = lr / (prob_arr[block_ii] * num_blocks);
            /// update model: x_hat = x_hat + c*A_S^T tmp_xty
            cblas_dgemv(CblasRowMajor, CblasTrans,
                        b, p, -2.0 * c, x_tr + block_ii * b * p, p, tmp_xty, 1,
                        1.0, x_hat, 1);
            /// hard threshold step
            hard_thresholding_operator(x_hat, sorted_ind, s, p);
            /// take the average of the model. (online to batch conversions.)
            cblas_dscal(p + 1, tt / (tt + 1.), x_bar, 1);
            cblas_daxpy(p + 1, 1 / (tt + 1.), x_hat, 1, x_bar, 1);
            tt++;
        }
        /// diverge cases because of the large learning rate: early stopping
        if (norm_l2(x_hat, p) >= 1e3) {
            break;
        }
        /// the error is sufficiently small.
        if (rooted_se <= tol_algo) {
            break;
        }
        // TODO check a local optimal.
        if (epoch_i >= 10) {
            /// we define it is a local optimal point if the previous 5
            /// epochs got the same losses and x_errors.
            bool is_local = true;
            double tmp_y_error = stat->epoch_losses[epoch_i];
            double tmp_x_error = stat->x_epoch_errors[epoch_i];
            for (int i = 1; i <= 5; i++) {
                if (tmp_y_error != stat->epoch_losses[epoch_i - i] ||
                    tmp_x_error != stat->epoch_losses[epoch_i - i]) {
                    is_local = false;
                    break;
                }
            }
            if (is_local) {
                break;
            }

        }
        if (verbose > 0) {
            printf("epoch %03d: x_error: %.6e y_error: %.6e\n",
                   epoch_i, stat->x_epoch_errors[epoch_i],
                   stat->epoch_losses[epoch_i]);
        }
    }
    /// to save results.
    for (int i = 0; i < p + 1; i++) {
        stat->x_hat[i] = x_hat[i];
        stat->x_bar[i] = x_bar[i];
    }
    stat->run_time = (double) (clock() - start_time) / CLOCKS_PER_SEC;
    free(sorted_ind);
    free(r_perm);
    free(tmp_xty);
    free(x_hat);
    free(x_bar);
    free(x_hat_tmp);
    return true;
}


