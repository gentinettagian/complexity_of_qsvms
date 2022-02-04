import numpy as np

class ShotBasedQuantumKernel():
    """
    Approximates a given Quantum Kernel by drawing shots from a Bernoulli distribution.
    """
    def __init__(self, kernel_matrix):
        self.K = kernel_matrix
    
    def approximate_kernel(self, shots, seed = 42, enforce_symmetry = True, batch_size = 1e9):
        M1, M2 = self.K.shape
        np.random.seed(seed)
        sample_means = []
        weights = []
        while shots > 0:
            if shots*M1*M2 > batch_size:
                batch = int(batch_size/(M1*M2))
            else:
                batch = shots
            shots = shots - batch
            weights.append(batch)

            rng = np.random.random((batch,M1,M2))
            samples = rng < self.K
            sample_mean = np.mean(samples,axis=0)
            if enforce_symmetry:
                sample_mean *= np.tri(*sample_mean.shape)
                sample_mean += np.tril(sample_mean,-1).T
            sample_means.append(sample_mean)
            
        
        approximate_kernel = np.sum(np.array(weights).reshape(-1,1,1)*np.array(sample_means),axis=0)/np.sum(weights)
        return approximate_kernel

class BinomialKernel():
    """
    Approximates a given Quantum Kernel with a binomial distribution.
    """
    def __init__(self, kernel_matrix):
        self.K = kernel_matrix
    
    def approximate_kernel(self, shots, seed = 42, enforce_symmetry = True, batch_size = 1e18, test_mode = False):
        np.random.seed(seed)

        sample_means = []
        weights = []

        while shots > 0:
            
            if shots > batch_size:
                batch = int(batch_size)
            else:
                batch = shots
                
            
            shots = shots - batch
            weights.append(batch)

            sample = np.random.binomial(batch, self.K) / batch
            

            if enforce_symmetry:
                sample *= np.tri(*sample.shape)
                sample += np.tril(sample,-1).T

            sample_means.append(sample)
            
    
        approximate_kernel = np.sum(np.array(weights).reshape(-1,1,1)*np.array(sample_means),axis=0)/sum(weights)
        if test_mode:
            return np.array(approximate_kernel, dtype = np.float64), len(weights)
        return np.array(approximate_kernel, dtype = np.float64)

if __name__ == "__main__":
    a = np.random.random((128,128))
    a *= np.tri(*a.shape)
    a += np.tril(a,-1).T
    K = BinomialKernel(a)
    exp = range(23)
    eps = np.zeros(23)
    for i in exp:
        print(i)
        a_prox = K.approximate_kernel(10**i)
        print(a_prox)
        eps[i] = np.linalg.norm(a - a_prox,ord=2)
    import matplotlib.pyplot as plt

    plt.plot(exp,eps)
    plt.yscale('log')
    plt.show()
