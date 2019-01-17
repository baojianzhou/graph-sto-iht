# -*- coding: utf-8 -*-
import os
import time
import pickle
from os import sys, path
import numpy as np
from itertools import product
from numpy.random import normal

import multiprocessing
from algo_wrapper import algo_sto_iht_wrapper

trial_i = 0
max_epochs, tol_algo, tol_rec = 500, 1e-7, 1e-6
selected_m = {4: [80, 140, 4, 5],
              8: [80, 100, 8, 9],
              12: [96, 144, 12, 14],
              20: [70, 75, 20, 21],
              28: [55, 60, 28, 29],
              36: [45, 50, 36, 37]}
root_p = '/network/rit/lab/ceashpc/bz383376/data/icml19/sparse_recovery/'
if not os.path.exists(root_p):
    os.mkdir(root_p)
if not os.path.exists(root_p + 'input/'):
    os.mkdir(root_p + 'input/')
root_input = root_p + 'input/simu/'
if not os.path.exists(root_input):
    os.mkdir(root_input)
root_output = root_p + 'output/'
if not os.path.exists(root_output):
    os.mkdir(root_output)
root_figs = root_p + 'figs/'
if not os.path.exists(root_figs):
    os.mkdir(root_figs)

lr, g, root, max_num_iter, verbose = 1.0, 1, -1, 50, 0
f_name = root_input + 'simu_test01_trial_%03d.pkl' % trial_i
all_data = pickle.load(open(f_name))
result_error = []
start_time = time.time()
for data_index, data in enumerate(all_data):
    x_tr, y_tr = data['x_tr'], data['y_tr']
    s, m, w, w0, b = data['s'], data['m'], data['w'], data['w0'], data['b']
    # x_tr, y_tr, max_epochs, lr, s, x_star, x0, tol_algo, b,
    #         with_replace=True, loss_func=0, verbose=0, lambda_l2=0.0,
    #         tol_rec=1e-6, use_py=False
    re = algo_sto_iht_wrapper(
        x_tr, y_tr, max_epochs, lr, s, w, w0, tol_algo, b,
        with_replace=True, loss_func=0, verbose=0, lambda_l2=0.0, tol_rec=1e-6,
        use_py=True)
    print(data_index, re['err'])
print('run_time: %.4e' % (time.time() - start_time))
