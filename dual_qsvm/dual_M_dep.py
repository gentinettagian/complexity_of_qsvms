from qiskit import Aer
from feature_maps import MediumFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.algorithms.classifiers import QSVC

from shot_based_kernel import ShotBasedQuantumKernel

from quantum_neural_networks import QuantumNeuralNetwork
from variational_forms import _VariationalForm
from feature_maps import MediumFeatureMap

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from my_svc import MySVC
from sklearn.svm import SVC

import pickle

from tqdm import tqdm

blue = '#1f77b4'
orange = '#ff7f0e'
green = '#2ca02c'
red = '#d62728'
violet = '#9467bd'
grey = '#7f7f7f'
cyan = '#17becf'

# other constants
lower_percentile = 0.159
upper_percentile = 0.841


# Function to generate artificial data
def get_data_generated(qnn, M=100, margin=0.1, bias=0, shuffle=True, seed=41, return_theta=False):
    """returns a toy dataset (binary classification) generated with respect to a specific quantum neural network, such 
    that the QNN can obtain 100% classification accuracy on the train set
    :param qnn: an instance of the QuantumNeuralNetwork class
    :param M: int, the desired size of the generated dataset
    :param margin: float in [-0.5, 0.5], the margin around 0.5 probability prediction where no data are included
    :param shuffle: bool, whether the data is ordered by class or shuffled
    :param seed: int, the random seed
    """
    rng = np.random.default_rng(seed)

    assert M % 2 == 0, 'M has to be even'

    # fix the variational form in the given QNN
    theta = rng.uniform(0, 2*np.pi, size=qnn.d)
    qnn.fit_theta(theta)

    class_0 = []
    class_1 = []

    # loop until the two lists both contain M/2 elements
    while len(class_0) < M//2 or len(class_1) < M//2:
        # generate a random point
        x = rng.uniform(0, 1, size=qnn.q)
        y_prob = qnn.predict_proba(np.array([x]))

        # strict class membership criteria if margin > 0
        criterion_0 = y_prob < (0.5 - margin/2 + bias)
        criterion_1 = (0.5 + margin/2 + bias) < y_prob

        # can only be true for a negative margin. Then randomly choose the class membership
        if criterion_0 and criterion_1:
            if np.random.choice([True, False]) and len(class_0) < M//2:
                class_0.append(x)
            elif len(class_1) < M//2:
                class_1.append(x)

        # class 0
        if criterion_0 and not criterion_1 and len(class_0) < M//2:
            class_0.append(x)

        # class 1
        if criterion_1 and not criterion_0 and len(class_1) < M//2:
            class_1.append(x)

    # generate the sorted X and y arrays
    y = np.zeros(M, dtype=int)
    y[M//2:] = 1
    X = np.array(class_0 + class_1)

    if shuffle:
        inds = rng.choice(M, M, replace=False)
        X = X[inds]
        y = y[inds]

    if return_theta:
        return X, y, 'generated', theta
    else:
        return X, y, 'generated'
    
# Adapt above method for QSVM (trivial QNN)
def generate_qsvm_data(feature_map,margin,M,M_test=None,seed=41):
    q = feature_map.q
    varform = _VariationalForm(q,1,None)
    qnn = QuantumNeuralNetwork(feature_map=feature_map,variational_form=varform)
    qnn.fit_theta(np.array([0]))
    X,y,_ = get_data_generated(qnn,M=M+M_test,margin=margin,seed=seed)
    y[y==0] = -1
    
    if M_test is None:
        return X, y
    else:
        return (X[:M], y[:M]), (X[M:], y[M:])

class DualExperiment():
    def __init__(self, margin, Cs = [10,1000], Ms = 2**np.arange(2,12), shots = 2**np.arange(2,20),
        qubits = 2, reps = 4, seed = 42):
        self.margin = margin
        self.Cs = Cs
        self.Ms = Ms
        self.shots = shots
        self.seed = seed
        self.qubits = qubits
        self.reps = reps

        self.feature_map = MediumFeatureMap(self.qubits, self.reps)
        

    def generate_data(self):
        max_M = 2048 #np.max(self.Ms)
        try:
            X = pd.read_csv(f'data/{self.qubits}-qubits/{self.margin}_X_{max_M}.csv')
            y = pd.read_csv(f'data/{self.qubits}-qubits/{self.margin}_y_{max_M}.csv')
        except:
            (X,y), _ = generate_qsvm_data(self.feature_map,self.margin,max_M,0,self.seed)
            X = pd.DataFrame(X)
            X.to_csv(f'data/{self.qubits}-qubits/{self.margin}_X_{max_M}.csv',index=False)
            y = pd.DataFrame(y)
            y.to_csv(f'data/{self.qubits}-qubits/{self.margin}_y_{max_M}.csv',index=False)
        
        self.X = np.array(X)
        self.y = np.array(y).reshape(-1)

        return X,y
    
    def load_data(self, M, seed = 42):
        assert (M <= 2048) & (M % 2 == 0)
        np.random.seed(seed)
        indices1 = np.random.randint(0,np.sum(self.y == 1),M//2)
        indices2 = np.random.randint(0,np.sum(self.y == -1),M//2)

        X1 = self.X[self.y == 1][indices1]
        y1 = self.y[self.y == 1][indices1]
        X2 = self.X[self.y == -1][indices2]
        y2 = self.y[self.y == -1][indices1]

        X = np.vstack([X1,X2])
        y = np.append(y1,y2)

        shuffle = np.random.choice(M, M, replace=False)
        return X[shuffle], y[shuffle]

    def run(self, seed = 42):
        self.generate_data()
        try:
            results = pd.read_csv(f'experiments/M_{self.margin}_data.csv')
        except:
            columns = ['seed','R','C','M','epsilon','epsilon_euclid1','epsilon_euclid2']
            results = pd.DataFrame(columns=columns)
            results.to_csv(f'experiments/M_{self.margin}_data.csv',index=False)

        for M in tqdm(self.Ms):
            print(f'M = {M}')
            X, y = self.load_data(M,seed)

            state_backend = QuantumInstance(Aer.get_backend('statevector_simulator'))
            state_kernel = QuantumKernel(feature_map=self.feature_map.get_reduced_params_circuit(), quantum_instance=state_backend)
            state_matrix = state_kernel.evaluate(X)

            for C in self.Cs:
                svc = SVC(kernel='precomputed',C=C)
                svc.fit(state_matrix,y)
                h_state = svc.decision_function(state_matrix)

                # Calculating noisy solution
                shots_based_kernel = ShotBasedQuantumKernel(state_matrix)

                for R in self.shots:
                    if ((results['seed'] == seed) & (results['R'] == R) & (results['C'] == C) & (results['M'] == M)).any():
                        continue
                    
                    R_shots_kernel = shots_based_kernel.approximate_kernel(R,seed)
                    svc_R = SVC(kernel='precomputed',C=C)
                    svc_R.fit(R_shots_kernel,y)
                    h_R = svc_R.decision_function(R_shots_kernel)
                    eps = np.max(np.abs(h_state - h_R))

                    eps_eucl1 = np.sum(np.abs(h_state - h_R)) / M # normalize
                    eps_eucl2 = np.linalg.norm(h_state - h_R,ord=2) / M # normalize

                    results.loc[results.shape[0]] = [seed, R, C, M, eps, eps_eucl1, eps_eucl2]
                    results.to_csv(f'experiments/M_{margin}_data.csv',index=False)

    
        
if __name__ == "__main__":

    Ms = 2**np.arange(2,9)
    shots = 2**np.arange(2,22)
    n_seeds = 5


    np.random.seed(43)
    seeds = np.random.randint(0,1e8,n_seeds)
    
   

    margin = -0.1
    experiment = DualExperiment(margin,Ms=Ms,shots=shots)
    for s in seeds:
        experiment.run(s)