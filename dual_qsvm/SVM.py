import numpy as np
import quadprog as qp

class SVM():
    """
    Implements classification using kernelizes support vector machines by solving the quadratic program
    in our paper. In particular, this uses l2-regularization for the slack variables, compared to the
    l1-normalization in sklearns SVC.
    """
    def __init__(self, kernel = 'precomputed', C = 10, verbose = False, cut_near_zeros = True) -> None:
        """
        C... regularization
        kernel... kernel method
        verbose... print stats
        cut_near_zeros... deletes weights with alpha < 1e-10
        """

        if kernel != 'precomputed':
            raise NotImplementedError('SVM currently only works for precomputed kernels.')
        self._C = C
        self._verbose = verbose
        self._cut_near_zeros = cut_near_zeros
    
    def fit(self, K, y):
        """
        Solves the quadratic program.
        K... Kernel matrix, shape (n, n)
        y... True labels, shape (n, )
        """
        Q = np.diag(y) @ K @ np.diag(y)
        
        # Setting up quadratic program as in the quadprog documentation
        # Minimize     1/2 x^T G x - a^T x
        # Subject to   C.T x >= b

        G = Q + 1/self._C * np.eye(len(y))
        a = np.ones(len(y))
        C = np.eye(len(y))
        b = np.zeros(len(y))

        try:
            sol = qp.solve_qp(G, a, C, b)
        except:
            return False
        self._alphas = sol[0]
        if self._cut_near_zeros:
            self._alphas[self._alphas < 1e-10] = 0

        self._support = self._alphas != 0
        self._dual_coef = self._alphas * y
        self._objective = sol[1]

        if self._verbose:
            print("Quadratic program solved")
            y_pred = self.predict(K)
            accuracy = np.sum(y == y_pred) / len(y)
            print("Training accuracy: ", accuracy)
            print("Objective: ", self._objective)
            print("Support vectors: ", np.sum(self._support))



        return True

    def decision_function(self, K):
        return (K @ self._dual_coef).reshape(-1)

    def predict(self, K):
        return np.sign(self.decision_function(K))
    


        