import numpy as np

class SVM():
    """
    Implements classification using kernelizes support vector machines by solving the quadratic program
    in our paper. In particular, this uses l2-regularization for the slack variables, compared to the
    l1-normalization in sklearns SVC.
    """
    def __init__(self, kernel = 'precomputed', C = 10) -> None:

        if kernel != 'precomputed':

            raise NotImplementedError
        