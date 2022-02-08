
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import pickle
from feature_maps import MediumFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit import Aer
from SVM import SVM
from shot_based_kernel import BinomialKernel


# constants
lower_percentile = 0.159
upper_percentile = 0.841

# colors
blue = '#1f77b4'
orange = '#ff7f0e'
green = '#2ca02c'
red = '#d62728'
violet = '#9467bd'
grey = '#7f7f7f'
cyan = '#17becf'

colors = [blue,orange,green,red,violet,grey,cyan]

plt.rcParams.update({'font.size': 24,
                     'xtick.labelsize': 20,
                     'ytick.labelsize': 20,
                     'axes.titlesize': 28,
                     'axes.labelsize': 28,
                     'mathtext.fontset': 'stix',
                     'font.family': 'STIXGeneral'})

class BinomialExperiment():
    """
    Experiment to determine M dependence of dual method.
    """
    def __init__(self, margin, C, qubits=2, reps=4, seed = 42, Ms = 2**np.arange(4,10), shots = 2**np.arange(4,11), 
        epsilons = np.array([0.001,0.005,0.01,0.05,0.1]), estimations = 3) -> None:
        self.feature_map = MediumFeatureMap(qubits, reps)
        self.margin = margin
        self.Ms = Ms
        self.seed = seed
        self.epsilons = np.array(epsilons)
        self.shots = shots
        self.estimations = estimations
        self.C = C

        backend = QuantumInstance(Aer.get_backend('statevector_simulator'))
        self.kernel = QuantumKernel(feature_map=self.feature_map.get_reduced_params_circuit(), quantum_instance=backend)
 
    def run(self):
        """
        1) For every M: 
        - given exact Kernel draw binomial samples until |h-h_R| < eps

        """

        
        self.minimal_R = np.zeros((len(self.Ms), self.estimations, len(self.epsilons)))
        try:
            results = pd.read_csv(f'experiments/binomial_experiment_{self.margin}_C_{self.C}.csv')
        except:
            results = pd.DataFrame(columns=['M', 'seed'] + [eps for eps in self.epsilons])
            results.to_csv(f'experiments/binomial_experiment_{self.margin}_C_{self.C}.csv',index=False)

        rng = np.random.default_rng(self.seed)
        seeds = rng.choice(10000, size=self.estimations, replace=False)

        for i, M in enumerate(self.Ms):
            # Exact kernel
            X, y = self.load_data(M,self.seed)
            K = self.kernel.evaluate(X)
            K[K > 1] = 1
            K[K < 0] = 0
            
            svc = SVM(kernel='precomputed',C=self.C)
            svc.fit(K,y)
            h_exact = svc.decision_function(K)

            
            for l in range(self.estimations):
                if np.any((results['M'] == M) & (results['seed'] == seeds[l])):
                    continue
                self.minimal_R[i,l,:] = self.get_R_for_eps(K, h_exact, y, seeds[l])
                results.loc[results.shape[0]] = [M, seeds[l]] + [R for R in self.minimal_R[i, l, :]]
            
                results.to_csv(f'experiments/binomial_experiment_{self.margin}_C_{self.C}.csv',index=False)

        return
        

    def load(self):
        results = pd.read_csv(f'experiments/binomial_experiment_{self.margin}_C_{self.C}.csv')

        self.minimal_R = np.zeros((len(self.Ms), self.estimations, len(self.epsilons)))
        for i, M in enumerate(self.Ms):
            for l in range(self.estimations):
                self.minimal_R[i,l,:] = np.array(results[results['M'] == M])[l,2:]
        return



            
    def plot(self):
        """
        2) Plot M ~ R(eps0,M) * #Kernel entries for all eps0
        3) Determine exponents
        """
        # total kernel evaluations
        kernel_evaluations = 0.5 * self.Ms*(self.Ms - 1)
        effective_R = self.minimal_R * kernel_evaluations.reshape(-1,1,1)

        exponents = np.zeros(len(self.epsilons))

        plt.figure(figsize=(10,7))
        for i, eps in enumerate(self.epsilons):
            means = np.median(effective_R[:,:,i],axis=-1)
            upper = np.quantile(effective_R[:,:,i],upper_percentile,axis=-1)
            lower = np.quantile(effective_R[:,:,i],lower_percentile,axis=-1)
            errors = np.array([means - lower, upper - means])

            p = np.polyfit(np.log(self.Ms), np.log(means), 1)
            exponents[i] = p[0]
            plt.errorbar(self.Ms, means, yerr=errors, marker='.', ecolor='grey', elinewidth=1., ls='',
            capsize=2, color=colors[i], ms=10, label = r'$\varepsilon = {{%s}}, \quad R \propto M^{{%.2f}}$'%(eps, p[0]))

            M_fine = np.geomspace(np.min(self.Ms),np.max(self.Ms))

            plt.plot(M_fine, np.exp(p[1])*M_fine**p[0],'-.',color=colors[i])

        plt.xscale('log')
        plt.yscale('log')
        plt.grid()
        plt.legend()
        plt.xlabel(r'Data size $M$')
        plt.ylabel(r'Total number of shots $R$')
        #plt.show()
        sep = 'separable' if self.margin > 0 else 'overlap'
        plt.savefig(f'plots/binomial_experiment_{sep}_C_{self.C}.png',dpi=300,bbox_inches='tight')

        return exponents
            
    
    def get_epsilon(self, h_exact, K_R, y):
        svc_R = SVM(kernel='precomputed',C=self.C)
        if svc_R.fit(K_R,y):
            h_R = svc_R.decision_function(K_R)
            return np.max(np.abs(h_R - h_exact))
        else:
            return 100.0
  
    def get_R_for_eps(self, K, h_exact, y, seed):
        R = 1024
        # Approximate kernel
        shots_kernel = BinomialKernel(K)
        K_start = shots_kernel.approximate_kernel(R, seed)
        start_eps = self.get_epsilon(h_exact, K_start, y)
           
        if start_eps < min(self.epsilons):
            print("The approximated Kernel is already accurate enough")
            # should not be the case
            return
        
        R_for_eps = np.zeros(len(self.epsilons))

        Rs_total = [R]
        eps_total = [start_eps]

        converged = False

        base = 2

        #plt.xscale('log')
        #plt.yscale('log')

        while not converged:
            R = base*R
            K_R, num_of_batches = shots_kernel.approximate_kernel(R, seed, test_mode=True)
            eps = self.get_epsilon(h_exact, K_R, y)
            R_for_eps[(R_for_eps == 0.) & (eps < self.epsilons)] = R


            Rs_total.append(R)
            eps_total.append(eps)
         
            if np.all(R_for_eps > 0):
                converged = True
            
            #if R > 1e10:
            #    base = 10
            
            #plt.scatter(R,eps)
            #plt.scatter(R,num_of_batches,marker='x')
            #plt.pause(0.05)

            
            
            
        
        
        
        #plt.show()
            
        return R_for_eps

        
  



    def load_data(self, M, seed = 42):
        assert (M <= 2048) & (M % 2 == 0)
        y = np.array(pd.read_csv(f'data/2-qubits/{self.margin}_y_2048.csv')).reshape(-1)
        X = np.array(pd.read_csv(f'data/2-qubits/{self.margin}_X_2048.csv'))

        np.random.seed(seed)
        indices1 = np.random.randint(0,np.sum(y == 1),M//2)
        indices2 = np.random.randint(0,np.sum(y == -1),M//2)

        X1 = X[y == 1][indices1]
        y1 = y[y == 1][indices1]
        X2 = X[y == -1][indices2]
        y2 = y[y == -1][indices1]

        X12 = np.vstack([X1,X2])
        y12 = np.append(y1,y2)

        shuffle = np.random.choice(M, M, replace=False)
        return X12[shuffle], y12[shuffle]


if __name__ == "__main__":
    epsilons = [0.001,0.01,0.1]
    for C in [10,1000]:
        for margin in [0.1,-0.1]:
            s = BinomialExperiment(margin,C,estimations=100, Ms = 2**np.arange(4,9), shots = 2**np.arange(4,12),epsilons=epsilons)
            s.run()
            #s.load()
            s.plot()
