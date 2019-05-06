## Stochastic IHT for Graph-structured Sparsity  Optimization

### Overview

Welcome to the repository of GraphStoIHT! This repository is only for 
reproducing all experimental results shown our paper. To install it via pip, 
please try [sparse-learn](https://github.com/baojianzhou/sparse-learn). 
Our work is due to several seminal works including 
[StoIHT](https://ieeexplore.ieee.org/abstract/document/8025727), 
[GraphCoSaMP](http://people.csail.mit.edu/ludwigs/papers/icml15_graphsparsity.pdf), 
and [AS-IHT](http://papers.nips.cc/paper/6483-fast-recovery-from-a-union-of-subspaces).
More details can be found in: "Baojian Zhou, Feng Chen, and Yiming Ying, 
Stochastic Iterative Hard Thresholding for Graph-structured Sparsity
Optimization, ICML, 2019".

------------------------------------------------------------------------------
This readme.txt
describes how to run GraphStoIHT and all baselines. The folder
--graph-sto-iht:
        contains all code ( including baselines), datasets, and
        results used in our paper.
--overlasso-package:
        download from [ http://cbio.ensmp.fr/˜ljacob/documents/
        overlasso-package.tgz ], which contains all l1/l2 mixed-norm
        methods.

Our code is written by Python and C code. I assume your Operating
System is GNU/Linux-based. However, if you have MacOS or MacBook, it will be
okay. The only dependencies of our programs is Python2.7 and GCC.


------------------------------------------------------------------------------
This section is to tell you how to prepare the environment. It has three steps:
    1.  install Python-2.7 and GCC (Linux/MacOS/MacBook already have them.)
    2.  install numpy, matplotlib (optional), and networkx (optional).
    3.  after the first two steps, run: 
            python setup.py build_ext --inplace
Step 3 will generate a sparse_module.so file. If you see this generated file,
then we are done for this section.


------------------------------------------------------------------------------
This section describes how to generate the exact (figures and tables) reported.

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
--4     means the number of cpus used for each program.
--0     means the start of the trial id.
--50    means the end of the trial id.
 
To generate results of Figure-2, run:
            python exp_sr_test02.py run_test 4
To generate results of Figure-3, run:
            python exp_sr_test03.py run_test 4 0 50
To generate results of Figure-4, run:
            python exp_sr_test01.py run_test 4 0 50
To generate results of Figure-5, run:
            python exp_sr_test06.py run_test 4 0 50
To generate results of Figure-6, run:
            python exp_sr_test04.py run_test 4 0 50
To generate results of Figure-7, run:
            python exp_sr_test05.py run_test 4 0 50
To generate results of Table 2, 3, 4, 5, run:
            python exp_bc_run.py run_test 0 20

Some programs above are time-cost if you only use 4 cpus. A better way is to
test them on 10 trials by replacing 50 with 10. After above steps, you should
be approximately reproduce our results reported in our paper. If you cannot
reproduce the results pleased email: --@--.


------------------------------------------------------------------------------
This section describes how to run l1/l2-mixed norm methods:
Download overlasso from:
http://cbio.ensmp.fr/˜ljacob/documents/overlasso-package.tgz. We downloaded a
version in overlasso-package. Run those l1/l2-mixed norm methods 20 times and
then generate the data and results in results folder.
