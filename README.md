------------------------------------------------------------------------------
Thank you for taking the time to review our code and datasets. This README.md 
file describes how to run our GraphStoIHT algorithm and all baselines. The 
folder graph-sto-iht contains all code ( including baselines), datasets, and 
results. I assume your Operating System is Linux/MacOS/MacBook.

------------------------------------------------------------------------------
This section is to tell you how to prepare the environment. 
You need to:
    0.  install Python-2.7 and GCC (Linux/MacOS/MacBook already have them.)
    1.  install numpy, matplotlib (optional), and networkx (optional).
    2.  run: 
            python setup.py build_ext --inplace

------------------------------------------------------------------------------
This section describes how to generate the exact(figures) reported.
To generate Figure-1, run:
            python exp_sr_test01.py gen_figures
To generate Figure-2, run:
            python exp_sr_test02.py show_test
To generate Figure-3, run:
            python exp_sr_test03.py show_test
To generate Figure-4, run:
            python exp_sr_test01.py show_test
To generate Figure-5, run:
            python exp_sr_test06.py show_test
To generate Figure-6, run:
            python exp_sr_test04.py show_test
To generate Figure-7, run:
            python exp_sr_test05.py show_test
To generate Table 2, 3, 4, 5, run:
            python exp_bc_run.py show_test
            
------------------------------------------------------------------------------
This section describes how to reproduce the results. In the following commands,
4 means the number of cpus used for each program.
 
To generate Figure-2, run:
            python exp_sr_test02.py run_test 4
To generate Figure-3, run:
            python exp_sr_test03.py run_test 4 0 50
To generate Figure-4, run:
            python exp_sr_test01.py run_test 4 0 50
To generate Figure-5, run:
            python exp_sr_test06.py run_test 4 0 50
To generate Figure-6, run:
            python exp_sr_test04.py run_test 4 0 50
To generate Figure-7, run:
            python exp_sr_test05.py run_test 4 0 50
To generate Table 2, 3, 4, 5, run:
            python exp_bc_run.py show_test

Some programs above, will spend a lot of time to training 
if you only use 4 cpus. A better way is to test them on 10 trials. Therefore,
all 50 could be change to 10.

 