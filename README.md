--------------------------------------------------------------------------------
Thank you for taking the time to review our code and datasets. This README.md 
file describes how to run our GraphStoIHT algorithm and all baselines. The 
folder graph-sto-iht contains all code, datasets, and results. I assume your
Operating System is Linux/MacOS/MacBook.

--------------------------------------------------------------------------------
Installation:  This part is to tell you how to prepare the environment. 
You need to:
    0.  install Python-2.7 and GCC (Linux/MacOS/MacBook already have them.)
    1.  install numpy, matplotlib (optional), and networkx (optional).
    2.  run: 
            python setup.py build_ext --inplace

--------------------------------------------------------------------------------

To generate Figure-1, run:
            python exp_sr_test01.py gen_figures
To show directly show Figure-2
