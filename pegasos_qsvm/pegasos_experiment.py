from qiskit import BasicAer
from feature_maps import MediumFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.datasets import ad_hoc_data
from pegasos import pegasos

from quantum_neural_networks import QuantumNeuralNetwork
from variational_forms import _VariationalForm
from feature_maps import MediumFeatureMap

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


plt.rcParams.update({'font.size': 24,
                     'xtick.labelsize': 20,
                     'ytick.labelsize': 20,
                     'axes.titlesize': 28,
                     'axes.labelsize': 20,
                     'mathtext.fontset': 'stix',
                     'font.family': 'STIXGeneral'})

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
    # Trivial variational form
    varform = _VariationalForm(q,1,None)
    # This qnn is equivalent to the QSVM
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

def run_experiment(margin,C,N,shots,M=1000,M_test=100,n_tests=100):
    """Runs experiments for the Pegasos algorithms and saves results to a csv file.
    margin.. Margin between the data classes. Positive for separable, negative for overlapping
    C.. Regulizer (1/lambda in the paper)
    N.. Number of iterations
    shots.. List of different number of shots 'R' to test
    M.. training data size
    M_test.. test data size
    n_tests.. number of tests
    """
    # Feature map for the experiment
    feature_map = MediumFeatureMap(2,4)

    # Creating artificial data
    (X,y), (Xt,yt) = generate_qsvm_data(feature_map,margin,M,M_test)
    
    # Checking whether experiment has already been partially done and loading existing data
    try:
        results = pd.read_csv(f'data/margin{margin}_data.csv')
    except:
        columns = ['seed','R','C','M']
        columns += [f'train acc. it. {n}' for n in range(N)]
        columns += [f'test acc. it. {n}' for n in range(N)]
        columns += [f'a error it. {n}' for n in range(N)]
        results = pd.DataFrame(columns=columns)
        results.to_csv(f'data/margin{margin}_data.csv',index=False)

    # Fix random seed to make reproducable
    np.random.seed(41)

    # Statevector backend as referecne to calcualte epsilon
    sv_backend = QuantumInstance(BasicAer.get_backend('statevector_simulator'))
    sv_kernel = QuantumKernel(feature_map=feature_map.get_reduced_params_circuit(), quantum_instance=sv_backend)

    # Kernel evaluation using statevector
    K = sv_kernel.evaluate(x_vec=X)
    K_test = sv_kernel.evaluate(x_vec=Xt,y_vec=X)

    # Generating Kernels for different number of shots using QASM simulator
    K_shots = np.zeros((len(shots),) + K.shape)
    print('Approximating Kernels')
    for i, R in tqdm(enumerate(shots)):
        R_shots_backend = QuantumInstance(BasicAer.get_backend('qasm_simulator'), shots=R,
                                seed_simulator=41, seed_transpiler=41)

        R_shots_kernel = QuantumKernel(feature_map=feature_map.get_reduced_params_circuit(), quantum_instance=R_shots_backend)
        K_shots[i] = R_shots_kernel.evaluate(x_vec=X)

    # Repeating experiment for 100 seeds
    seeds = np.random.randint(1,1e5,n_tests)
    
    for s in tqdm(seeds):
        algorithm_globals.random_seed = s
        # Run pegasos with statevector as reference
        _, a, _ = pegasos(K,y,N,C,seed=s,full_returns=True)

        for i, R in enumerate(shots):
            # Check whether this has already been calculated
            if ((results['seed'] == s) & (results['R'] == s) & (results['C'] == C)).any():
                continue

            # Run pegasos with finite shots 'R'
            y2,a2,_,_ = pegasos(K_shots[i],y,N,C,mu=0,seed=s,full_returns=True)
            # Calculating the errors on the weights
            errors_a = np.linalg.norm(a - a2,axis = 1,ord = 1)
            # Calculating accuracy on test and training set
            accuracies = accuracy(y2,y)
            accuracies_test = np.zeros_like(accuracies)
            for n in range(N):
                y_n = np.sign((a2[n,:] * y) @ K_test.T)
                accuracies_test[n] = accuracy(y_n,yt)

            # Saving results to csv
            results.loc[results.shape[0]] = [s, R, C, M] + accuracies.tolist() + accuracies_test.tolist() + errors_a.tolist()
            results.to_csv(f'experiments/shots_margin{margin}_data.csv',index=False)

def create_plots(filename,N):
    """
    Loads the data from 'filename' and creates plots for training and test accuracy, as well as 
    plots showing the evolution of the error on the weights alpha
    """
    # Load data
    data = pd.read_csv(filename)
    # Get list of the number of shots used
    shots = list(set(data['R']))
    shots.sort()
    # Get list of the regulizers used
    Cs = list(set(data['C']))
    cols = [blue,orange,green,violet]
    # x-axis showing the number of iterations
    x = np.arange(N)
    # Figure for training accuracy plots
    fig_acc, axs_acc = plt.subplots(len(Cs),sharex=True,figsize=[12,12])
    # Figure for test accuracy plots
    fig_tacc, axs_tacc = plt.subplots(len(Cs),sharex=True,figsize=[12,12])
    # Figure for error evolution plots
    fig_a, axs_a = plt.subplots(len(Cs),sharex=True,figsize=[12,12])

    for j, C in enumerate(Cs):
        for i, R in enumerate(shots):
            acc = np.array(data.loc[(data['R'] == R) & (data['C'] == C)].iloc[:,4:4 + N])
            if acc.shape[0] is not 0:
                acc_mean = np.mean(acc,axis=0)
                acc_lower = np.quantile(acc, lower_percentile, axis=0)
                acc_upper = np.quantile(acc, upper_percentile, axis=0)
                axs_acc[j].plot(x, acc_mean[:N], label=f'$R={int(R)}$')
                axs_acc[j].fill_between(x, acc_lower[:N], acc_upper[:N], alpha=0.3, edgecolor=None)

            tacc = np.array(data.loc[(data['R'] == R) & (data['C'] == C)].iloc[:,4 + N:4 + 2*N])
            if tacc.shape[0] is not 0:
                tacc_mean = np.mean(tacc,axis=0)
                tacc_lower = np.quantile(tacc, lower_percentile, axis=0)
                tacc_upper = np.quantile(tacc, upper_percentile, axis=0)
                axs_tacc[j].plot(x, tacc_mean[:N], label=f'$R={int(R)}$')
                axs_tacc[j].fill_between(x, tacc_lower[:N], tacc_upper[:N], alpha=0.3, edgecolor=None)

            a = np.array(data.loc[(data['R'] == R) & (data['C'] == C)].iloc[:,4 + 2*N:])
            if a.shape[0] is not 0:
                a_mean = np.mean(a,axis=0)
                a_lower = np.quantile(a, lower_percentile, axis=0)
                a_upper = np.quantile(a, upper_percentile, axis=0)
                axs_a[j].plot(x, (a_mean/C)[:N], label=f'$R={int(R)}$')
                axs_a[j].fill_between(x, (a_lower/C)[:N], (a_upper/C)[:N], alpha=0.3, edgecolor=None)

        axs_acc[j].grid()
        axs_tacc[j].grid()
        axs_a[j].grid()
        
        # Hard coded labels for the paper
        if C == 1000:
            axs_acc[j].set(ylabel=fr'Training accuracy, $\lambda = {1/C}$')
            axs_tacc[j].set(ylabel=fr'Test accuracy, $\lambda = {1/C}$')
            axs_a[j].set(ylabel=r'$\lambda\, ||\mathbf{\alpha}_R - \mathbf{\alpha}||,\quad \lambda = 0.001$')
        elif C == 10:
            axs_acc[j].set(ylabel=fr'Training accuracy, $\lambda = {1/C}$',xlabel='Iterations')
            axs_tacc[j].set(ylabel=fr'Test accuracy, $\lambda = {1/C}$',xlabel='Iterations')
            axs_a[j].set(ylabel=r'$\lambda\, ||\mathbf{\alpha}_R - \mathbf{\alpha}||,\quad \lambda = 0.1$',xlabel='Iterations')

        axs_a[0].legend(loc='upper right')
        axs_tacc[1].legend(loc='lower right')
        axs_acc[1].legend(loc='lower right')
        
    fig_acc.savefig(filename[:-8] + f'acc_plot{N}.png',dpi=300,bbox_inches='tight')
    fig_tacc.savefig(filename[:-8] + f'tacc_plot{N}.png',dpi=300,bbox_inches='tight')
    fig_a.savefig(filename[:-8] + f'a_plot{N}.png',dpi=300,bbox_inches='tight')
        

   
        

if __name__ == "__main__":
    '''
    shots = [1,2,4,8,64,256,512,1024]
    margin = 0.1
    C = 10.
    N = 500
    M = 100
    M_test = 20
    run_experiment(margin,C,N,shots,M,M_test,n_tests=100)
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
    """
    mus = [0,0.01,0.05,0.1,0.2]
    margin = -0.1
    C = 1000.
    N = 500
    M = 1000
    M_test = 100
    run_experiment(margin,C,N,mus,M,M_test)
    mus = [0,0.01,0.05,0.1,0.2]
    margin = 0.1
    C = 10.
    N = 500
    M = 1000
    M_test = 100
    run_experiment(margin,C,N,mus,M,M_test)
    """
    """
    mus = [0.5]
    margin = -0.1
    C = 10.
    N = 500
    M = 1000
    M_test = 100
    run_experiment(margin,C,N,mus,M,M_test)
    mus = [0.5]
    margin = -0.1
    C = 1000.
    N = 500
    M = 1000
    M_test = 100
    run_experiment(margin,C,N,mus,M,M_test)
    """

    create_plots('experiments/shots_margin-0.1_data.csv',500,zoom=False)
    create_plots('experiments/shots_margin0.1_data.csv',500,zoom=False)