from sklearn.svm import SVC
import numpy as np

class MySVC(SVC):
    """
    decision function fixed
    """
    def __init__(self,kernel,C):
        super().__init__(kernel=kernel,C=C)
    
    def decision_function(self, K, y):
        alphas = self.dual_coef_
        supp = self.support_
        return np.dot(alphas*y[supp],K[supp,:])
