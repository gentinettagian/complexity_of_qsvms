from qiskit import BasicAer
from feature_maps import MediumFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.algorithms.classifiers import QSVC



from quantum_neural_networks import QuantumNeuralNetwork
from variational_forms import _VariationalForm
from feature_maps import MediumFeatureMap

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def accuracy(y1,y2):
    return (1. - np.sum(np.abs(y1 - y2),axis=-1)/(2.*len(y1)))



def run_experiment(margin,C,shots,M=100,M_test=10,n_tests=100):

    # Feature map for the experiment
    feature_map = MediumFeatureMap(2,4)

    
    
    # Checking whether experiment has already been partially done and loading existing data
    try:
        results = pd.read_csv(f'experiments/margin_{margin}_data.csv')
    except:
        columns = ['seed','R','C','M','epsilon']
        results = pd.DataFrame(columns=columns)
        results.to_csv(f'experiments/margin_{margin}_data.csv',index=False)

    np.random.seed(41)
    # Repeating experiment for 100 seeds
    seeds = np.random.randint(1,1e5,n_tests)

    
    for s in tqdm(seeds):
        algorithm_globals.random_seed = s
        np.random.seed(s)

        # Creating artificial data
        (X,y), (Xt,yt) = generate_qsvm_data(feature_map,margin,M,M_test,seed=s)

        state_backend = QuantumInstance(BasicAer.get_backend('statevector_simulator'))

        state_kernel = QuantumKernel(feature_map=feature_map.get_reduced_params_circuit(), quantum_instance=state_backend)

        qsvc = QSVC(quantum_kernel=state_kernel,C=C)
        qsvc.fit(X,y)
        h_state = qsvc.decision_function(X)

        # Generating Kernels for different number of shots
        print('Approximating Kernels')
        for i, R in tqdm(enumerate(shots)):
            # Check whether this has already been calculated
            if ((results['seed'] == s) & (results['R'] == R) & (results['C'] == C) & (results['M'] == M)).any():
                continue

            R_shots_backend = QuantumInstance(BasicAer.get_backend('qasm_simulator'), shots=int(R),
                                    seed_simulator=s, seed_transpiler=s)

            R_shots_kernel = QuantumKernel(feature_map=feature_map.get_reduced_params_circuit(), quantum_instance=R_shots_backend)
            qsvc_R = QSVC(quantum_kernel=R_shots_kernel,C=C)
            qsvc_R.fit(X,y)
            h_R = qsvc_R.decision_function(X)
            eps = np.max(np.abs(h_state - h_R))

            results.loc[results.shape[0]] = [s, R, C, M, eps]
            results.to_csv(f'experiments/margin_{margin}_data.csv',index=False)

   

if __name__ == "__main__":

    Ms = [50] #np.arange(2,20,2)
    shots = [4,8,16,32,64,128,256,512,1024,2048,4096] #np.geomspace(10,10000,100,dtype=int)
    N = 2000
    M_test = 0

    margin = 0.1
    C = 10.
    for M in Ms:
        run_experiment(margin,C,shots,M,M_test,n_tests=10)

    margin = -0.1
    C = 10.
    for M in Ms:
        run_experiment(margin,C,shots,M,M_test,n_tests=10)
    
    margin = -0.1
    C = 1000.
    for M in Ms:
        run_experiment(margin,C,shots,M,M_test,n_tests=10)

    margin = 0.1
    C = 1000.
    for M in Ms:
        run_experiment(margin,C,shots,M,M_test,n_tests=10)

    

    '''
    shots = [1,2,4,8,64,256,512,1024]
    margin = 0.1
    C = 1000.
    N = 500
    M = 100
    M_test = 20
    run_experiment(margin,C,N,shots,M,M_test,n_tests=100)
    shots = [1,2,4,8,64,256,512,1024]
    margin = -0.1
    C = 10.
    N = 500
    M = 100
    M_test = 20
    run_experiment(margin,C,N,shots,M,M_test,n_tests=100)
    shots = [1,2,4,8,64,256,512,1024]
    margin = -0.1
    C = 1000.
    N = 500
    M = 100
    M_test = 20
    run_experiment(margin,C,N,shots,M,M_test,n_tests=100)
    '''
   

    #create_plots('experiments/shots_margin-0.1_data.csv',500,zoom=False)
    #create_plots('experiments/margin_0.1_data.csv',500,zoom=False)