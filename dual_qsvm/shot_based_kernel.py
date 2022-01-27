import numpy as np

class ShotBasedQuantumKernel():
    """
    Approximates a given Quantum Kernel by drawing shots from a Bernoulli distribution.
    """
    def __init__(self, kernel_matrix):
        self.K = kernel_matrix
    
    def approximate_kernel(self, shots, seed = 42):
        M1, M2 = self.K.shape
        np.random.seed(seed)
        rng = np.random.random((shots,M1,M2))
        samples = rng < self.K
        sample_mean = np.mean(samples,axis=0)
        return sample_mean

