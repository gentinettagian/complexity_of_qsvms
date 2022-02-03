import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import pickle
from feature_maps import MediumFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit import Aer
from sklearn.svm import SVC
from shot_based_kernel import ShotBasedQuantumKernel


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

class SmartExperiment():
    """
    Experiment to determine M dependence of dual method.
    """
    def __init__(self, margin, qubits=2, reps=4, seed = 42, Ms = 2**np.arange(4,10), shots = 2**np.arange(4,11), 
        epsilons = np.array([0.001,0.005,0.01,0.05,0.1]), estimations = 3) -> None:
        self.feature_map = MediumFeatureMap(qubits, reps)
        self.margin = margin
        self.Ms = Ms
        self.seed = seed
        self.epsilons = epsilons
        self.shots = shots
        self.estimations = estimations

        backend = QuantumInstance(Aer.get_backend('statevector_simulator'))
        self.kernel = QuantumKernel(feature_map=self.feature_map.get_reduced_params_circuit(), quantum_instance=backend)
 
    def run(self):
        """
        1) For every M: 
        - Create error model for how||K - K_R|| scales with M and R
        - Interpolate K(t) = K*t + (1 - t)*K_R
        - Determine epsilon as a function of t
        - For eps0 find smallest t s.t. epsilon(t) < eps0
        - Determine ||K - K(t)||
        - Use error model to find R(eps0,M)

        """

        
        self.minimal_R = np.zeros((len(self.Ms), self.estimations, len(self.epsilons)))
        
        results = pd.DataFrame(columns=['M', 'seed'] + [eps for eps in self.epsilons])
        results.to_csv(f'experiments/smart_experiment_{self.margin}.csv',index=False)

        np.random.seed(self.seed)
        seeds = np.random.randint(0,1000,self.estimations)

        for i, M in enumerate(self.Ms):
            # Exact kernel
            X, y = self.load_data(M,self.seed)
            K = self.kernel.evaluate(X)

            svc = SVC(kernel='precomputed',C=10,random_state=42)
            svc.fit(K,y)
            h_exact = svc.decision_function(K)

            # Approximate kernel
            shots_kernel = ShotBasedQuantumKernel(K)
            kernel_approximations = np.zeros((len(self.shots),self.estimations) + K.shape)
            kernel_errors = np.zeros((len(self.shots),self.estimations))

            for j, R in enumerate(self.shots):
                for l in range(self.estimations):
                    kernel_approximations[j,l,:] = shots_kernel.approximate_kernel(R, seeds[l])
                    kernel_errors[j,l] = np.linalg.norm(kernel_approximations[j,l,:] - K, ord=2) # Operator norm
            
            # Create error model
            error_means = np.mean(kernel_errors,axis=-1)
            p = np.polyfit(np.log(self.shots),np.log(error_means),1)
            def shots_for_kernel_error(error):
                return error**(1/p[0])/np.exp(p[1]/p[0])
            
            
            # Uncomment to see plots of error model
            '''
            plt.errorbar(error_means, shots, xerr = np.std(kernel_errors,axis=-1), ls = '', capsize=2)
            fine_err = np.linspace(np.min(error_means),np.max(error_means))
            plt.plot(fine_err, shots_for_kernel_error(fine_err),'--',color='grey')
            plt.xscale('log')
            plt.yscale('log')
            plt.show()
            '''
            
            for l in range(self.estimations):
                errors_for_eps = self.get_error_for_eps(self.epsilons, K, kernel_approximations[-1,l,:], h_exact, y)
                self.minimal_R[i, l, :] = np.ceil(shots_for_kernel_error(errors_for_eps))
                results.loc[results.shape[0]] = [M, seeds[l]] + [R for R in self.minimal_R[i, l, :]]
            
            results.to_csv(f'experiments/smart_experiment_{self.margin}.csv',index=False)

        return
        

    def load(self):
        results = pd.read_csv(f'experiments/smart_experiment_{self.margin}.csv')

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
            means = np.mean(effective_R[:,:,i],axis=-1)
            upper = np.quantile(effective_R[:,:,i],upper_percentile,axis=-1)
            lower = np.quantile(effective_R[:,:,i],lower_percentile,axis=-1)
            errors = np.array([means - lower, upper - means])
            plt.errorbar(self.Ms, means, yerr=errors, marker='.', ecolor='grey', elinewidth=1., ls='',
            capsize=2, color=colors[i], ms=10, label = f'{eps}')

            p = np.polyfit(np.log(self.Ms), np.log(means), 1)
            exponents[i] = p[0]

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
        plt.savefig(f'plots/smart_experiment_{sep}.png',dpi=300,bbox_inches='tight')

        return exponents
            
    
    def get_epsilon(self, h_exact, K_R, y):
        svc_R = SVC(kernel='precomputed',C=10,random_state=42)
        svc_R.fit(K_R,y)
        h_R = svc_R.decision_function(K_R)
        return np.max(np.abs(h_R - h_exact))
  
    def get_error_for_eps(self, K, K_R, h_exact, y):
        start_eps = self.get_epsilon(h_exact, K_R, y)
        if start_eps < min(self.epsilons):
            print("The approximated Kernel is already accurate enough")
            # should not be the case
            return
        
        error_for_eps = np.zeros(len(self.epsilons))
        K_t = lambda t: K*t + (1 - t)*K_R

        t_start, t_end = 1e-5, 1
        converged = False
        ts_total = []
        eps_total = []
        while not converged:
            ts = (1 - np.geomspace(t_start,t_end,100))[::-1]

            for t in ts:
                ts_total.append(t)
                eps = self.get_epsilon(h_exact, K_t(t), y)
                eps_total.append(eps)
                error_for_eps[(error_for_eps == 0.) & (eps < self.epsilons)] = np.linalg.norm(K - K_t(t), ord=2)
                if np.all(error_for_eps > 0):
                    converged = True
                    break
            
            t_end = t_start
            t_start = t_start**2
        
        #plt.plot(ts_total,eps_total)
        #plt.show()
            
        return error_for_eps

        
  



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
    s = SmartExperiment(-0.1,estimations=10, Ms = 2**np.arange(4,13), shots = 2**np.arange(4,12))
    s.run()
    s.plot()
    s = SmartExperiment(0.1,estimations=10, Ms = 2**np.arange(4,13), shots = 2**np.arange(4,12))
    s.run()
    s.plot()