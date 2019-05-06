## Stochastic IHT for Graph-structured Sparsity  Optimization

### Overview

Welcome to the repository of GraphStoIHT! This repository is only for 
reproducing all experimental results shown our paper. To install it via pip, 
please try [sparse-learn](https://github.com/baojianzhou/sparse-learn). 
Our work is due to several seminal works including 
[StoIHT](https://ieeexplore.ieee.org/abstract/document/8025727), 
[GraphCoSaMP](http://people.csail.mit.edu/ludwigs/papers/icml15_graphsparsity.pdf), 
and [AS-IHT](http://papers.nips.cc/paper/6483-fast-recovery-from-a-union-of-subspaces).
More details of GraphStoIHT can be found in: "Baojian Zhou, Feng Chen, and Yiming Ying, 
Stochastic Iterative Hard Thresholding for Graph-structured Sparsity
Optimization, ICML, 2019".

Our code is written by Python and C11. The implementation of head and tail 
projection are almost directly from Dr. Ludwig's excellent code
[cluster_approx](https://github.com/ludwigschmidt/cluster_approx), 
[graph_sparsity](https://github.com/ludwigschmidt/graph_sparsity_experiments) and
[pcst_fast](https://github.com/fraenkel-lab/pcst_fast). The reason I used C11 
to reimplement these two projections is to hope to get a little bit faster speed 
by using continuous memory (but the improvement is not significant at all!). I 
assume your Operating System is GNU/Linux-based( a GCC compiler is needed!). 
However, if you have MacOS or MacBook, it will be okay. The only dependencies 
of our programs is Python2.7 and GCC.

### Instructions
This section describes how to run GraphStoIHT and all baselines. 
The $\ell_1$ norm-based method is downloaded from 
[OverLasso](http://lbbe.univ-lyon1.fr/-Jacob-Laurent-.html?lang=fr). 
It contains all l1/l2 mixed-norm methods. This section is to tell you how to 
prepare the environment. It has three steps:

    1.  install Python-2.7 and GCC (Linux/MacOS/MacBook already have them.)

    2.  install numpy, matplotlib (optional), and networkx (optional).

    3.  after the first two steps, run: 
            python setup.py build_ext --inplace
Step 3 will generate a sparse_module.so file. If you see this generated file,
then we are done for this section.


### Graph Sparse Linear Regression

This section describes how to generate the exact figures reported in our paper.

##### 1. Figure 1

To generate Figure-1, run:

```sh
$ python exp_sr_test01.py gen_figures
```
Four figures will be generated in results folder. We show the four generated figures
in the following.

Graph with sparsity 8 |  Graph with sparsity 20 |  Graph with sparsity 28 |  Graph with sparsity 36
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------|
<img src="results/results_exp_sr_test01_s_08.png" width="200" height="200"> |  <img src="results/results_exp_sr_test01_s_20.png" width="200" height="200"> | <img src="results/results_exp_sr_test01_s_28.png" width="200" height="200"> |  <img src="results/results_exp_sr_test01_s_36.png" width="200" height="200"> 


##### 2. Figure 2

To show Figure-2, run:

```sh
$ python exp_sr_test02.py show_test
```
To reproduce Figure-2, run:

```sh
$ python exp_sr_test02.py run_test 4
```
Here, the parameter 4 is the number of CPU will be used. The left illustrates 
the estimation error as a function of epochs for different choices of b. When 
b= 180, it degenerates to GraphIHT (the dashed line).  The right part 
shows the estimation error as a function of iterations for different choices of η. 

Choice of b and η.|
:-------------------------:|
<img src="results/results_exp_sr_test02.png" width="800" height="300">|

##### 3. Figure 3

To show Figure-3, run:
            
```sh
$ python exp_sr_test03.py show_test
```
To reproduce Figure-3, run:

```sh
$ python exp_sr_test03.py run_test 4 0 50
```
Here, 4 is the number of CPUs used for each program. 0 is the start 
of the trial id. 50 is the end of the trial id. It means we try to run 
50 trials and then take the average.

|  |      Choice of b and η.      |  |
|----------|:-------------:|------|
|  &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;| <img src="results/results_exp_sr_test03.png" width="600" height="300"> | &emsp;&emsp;&emsp;&emsp;&emsp;&emsp; | 


##### 4. Figure 4

To show Figure-4, run:
            
```sh
$ python exp_sr_test01.py show_test
```
To reproduce Figure-4, run:

```sh
$ python exp_sr_test01.py run_test 4 0 50
```
Here, 4 is the number of CPUs used for each program. 0 is the start 
of the trial id. 50 is the end of the trial id. It means we try to run 
50 trials and then take the average.

|  |      Probability of recovery on synthetic dataset.      |  |
|----------|:-------------:|------|
|  &emsp;&emsp;&emsp;&emsp;&emsp;| <img src="results/results_exp_sr_test01.png" width="650" height="450"> | &emsp;&emsp;&emsp;&emsp;&emsp; |

Probability of recovery on synthetic dataset. The probability of recovery as 
function of the number of observations m.

##### 5. Figure 5

To show Figure-5, run:
            
```sh
$ python exp_sr_test06.py show_test
```
To reproduce Figure-5, run:

```sh
$ python exp_sr_test06.py run_test 4 0 50
```

|  |      Probability of recovery on three 50×50 resized real im-ages.      |  |
|----------|:-------------:|------|
|  &emsp;&emsp;&emsp;&emsp;&emsp;| <img src="results/results_exp_sr_test06.png" width="650" height="400"> | &emsp;&emsp;&emsp;&emsp;&emsp; |

            
##### 6. Figure 6

To show Figure-6, run:
            python exp_sr_test04.py show_test

To generate results of Figure-6, run:
            python exp_sr_test04.py run_test 4 0 50

|  |      The probability of recovery as a function of oversampling ratio.      |  |
|----------|:-------------:|------|
|  &emsp;&emsp;&emsp;&emsp;&emsp;| <img src="results/results_exp_sr_test04.png" width="650" height="450"> | &emsp;&emsp;&emsp;&emsp;&emsp; |

he oversampling ratio is defined as the number of observations m divided by sparsity s, i.e.,m/s. These four public benchmark graphs (a), (b), (c), and (d) in the upper row are from Arias-Castro et al.(2011)

            
##### 7. Figure 7
To show Figure-7, run:
            python exp_sr_test05.py show_test
            
To generate results of Figure-7, run:
            python exp_sr_test05.py run_test 4 0 50
            
|  |      The probability of recovery as a function of oversampling ratio.      |  |
|----------|:-------------:|------|
|  &emsp;&emsp;&emsp;&emsp;&emsp;| <img src="results/results_exp_sr_test05.png" width="650" height="350"> | &emsp;&emsp;&emsp;&emsp;&emsp; |

##### 8. Figure 8
To generate Table 2, 3, 4, 5, run:
            python exp_bc_run.py show_test

            
------------------------------------------------------------------------------
This section describes how to reproduce the results. In the following commands,
--4     means the number of cpus used for each program.
--0     means the start of the trial id.
--50    means the end of the trial id.


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
