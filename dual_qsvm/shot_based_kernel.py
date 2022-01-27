import numpy as np

class ShotBasedQuantumKernel():
    """
    Approximates a given Quantum Kernel by drawing shots from a Bernoulli distribution.
    """
    def __init__(self, kernel_matrix):
        self.K = kernel_matrix
    
    def approximate_kernel(self, shots, seed = 42, enforce_symmetry = True):
        M1, M2 = self.K.shape
        np.random.seed(seed)
        rng = np.random.random((shots,M1,M2))
        samples = rng < self.K
        sample_mean = np.mean(samples,axis=0)
        if enforce_symmetry:
            sample_mean *= np.tri(*sample_mean.shape)
            sample_mean += np.tril(sample_mean,-1).T
        return sample_mean

