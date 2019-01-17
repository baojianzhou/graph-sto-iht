#ifndef ALGO_STO_IHT_H
#define ALGO_STO_IHT_H

#include <stdlib.h>
#include <stdbool.h>

typedef enum LossFunc {
    LeastSquare,
    Logistic
} LossFunc;

typedef struct {
    double *x_tr;       //  training data samples (n,p) (must specify   )
    double *y_tr;       //  labels/measurements  (n,1)  (must specify   )
    int n;              //  sample dimension            (must specify   )
    int p;              //  feature dimension           (must specify   )
    int s;              //  sparsity parameter          (must specify   )
    int b;              //  the block size              (must specify   )
    LossFunc loss_func; //  loss function chosen        (must specify   )
    double lr;          //  learning rate               (default is 1.0 )
    double *w0;         //  initial point (p,1)         (default is 0v  )
    int max_epochs;     //  maximal # of iterations     (default is 500 )
    double tol_algo;    //  the tolerance of the algo   (default is 1e-7)
    double tol_rec;     //  the tolerance of recovery   (default is 1e-6)
    bool with_replace;  //  stochastic with replacement (default is True)
    int verbose;        //  verbose level               (default is 0   )
    double *x_star;     //  true w(sparse recovery)     (default is null)
    double *prob_arr;   //  probability array of blocks (default is DUD )
    double lambda_l2;   //  l2 regularization parameter (default is 0.0 )
} StoIHTPara;

typedef struct {
    double *x_hat;          //  result x_hat
    double *x_bar;          //  result x_bar
    double *epoch_losses;   //  list of losses for each single epoch.
    double *iter_losses;    //  list of losses for each single iteration.
    double *x_epoch_errors; //  list of errors for each single epoch.
    double *x_iter_errors;  //  list of errors for each single iteration.
    int num_epochs;         //  number of epochs used.
    double run_time;        //  run time of the algorithm
    double run_time_proj;   //  run time of the projection operator
} ReStoIHTStat;

ReStoIHTStat *make_re_sto_iht_stat(int p, int b, int max_epochs);

bool free_re_sto_iht_stat(ReStoIHTStat *re_stat);

bool algo_sto_iht_logit(StoIHTPara *para, ReStoIHTStat *stat);

bool algo_sto_iht_least_square(StoIHTPara *para, ReStoIHTStat *stat);

#endif //ALGO_STO_IHT_H

