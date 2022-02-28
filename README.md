This repository contains all the code to reproduce the data and figures for the paper _The complexity of quantum support vector machines_ by Gian Gentinetta, Arne Thomsen, David Sutter and Stefan Woerner ([WIP](WIP)).


Setup
=====
To reproduce the experiments, I highly suggest you create a new Python3 environment (e.g. with the `venv` tool). Python3.9 was used for the paper, though other versions of Python3 will probably also work fine. In your new environment install following packages (the exact version numbers are important!):
* numpy 1.21.4
* pandas 1.3.4
* quadprog 0.1.11
* tqdm 4.62.3
* scikit-learn 1.0.1
* qiskit-aer 0.9.1
* qiskit-ignis 0.6.0
* qiskit-ibmq-provider 0.18.0
* qiskit-aqua 0.9.5

In addition, the `qiskit-machine-learning` and `qiskit-terra` packages have to be cloned from the `complexity_of_qsvm` branches of the repositories

* https://github.com/gentinettagian/qiskit-machine-learning.git
* https://github.com/gentinettagian/qiskit-terra.git

and added separately using `pip install -e`.

If you wish to regenerate the plots, you will additionaly require matplotlib and latex to be installed.



Overview of code
================
The directory is divided into three folders for the three different models discussed in the paper. `dual_qsvm` contains the code to generate the figures in Section 4.2. `pegasos_qsvm` contains the code to generate the figures in Section 4.1, 4.3 and Appendix B. `approx_qsvm` contains the code to generate the figures in Section 4.4.