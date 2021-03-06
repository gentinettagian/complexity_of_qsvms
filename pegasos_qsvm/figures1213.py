from qiskit import Aer
from feature_maps import MediumFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.kernels import QuantumKernel
from pegasos import pegasos

from shot_based_kernel import ShotBasedQuantumKernel

from quantum_neural_networks import QuantumNeuralNetwork
from variational_forms import _VariationalForm
from feature_maps import MediumFeatureMap


import pickle

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
    if len(y1.shape) == 2:
        n = y1.shape[1]
    else:
        n = len(y1)
    return np.sum(y1 == y2,axis=-1)/n

def hinge_loss(y_true,y_pred):
    loss = 1. - y_true*y_pred
    loss[loss < 0 ] = 0.0
    return loss


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
        results = pd.read_csv(f'data/pegasos_{margin}_data{N}.csv')
    except:
        columns = ['seed','R','C','M']
        columns += [f'train acc. it. {n}' for n in range(N)]
        columns += [f'test acc. it. {n}' for n in range(N)]
        columns += [f'a error it. {n}' for n in range(N)]
        results = pd.DataFrame(columns=columns)
        results.to_csv(f'data/pegasos_{margin}_data{N}.csv',index=False)

    # Fix random seed to make reproducable
    np.random.seed(41)

    # Statevector backend as referecne to calcualte epsilon
    sv_backend = QuantumInstance(Aer.get_backend('statevector_simulator'))
    sv_kernel = QuantumKernel(feature_map=feature_map.get_reduced_params_circuit(), quantum_instance=sv_backend)

    # Kernel evaluation using statevector
    K = sv_kernel.evaluate(x_vec=X)
    K_test = sv_kernel.evaluate(x_vec=Xt,y_vec=X)

    # Generating Kernels for different number of shots
    shots_kernel = ShotBasedQuantumKernel(K)

    # Repeating experiment for 100 seeds
    seeds = np.random.randint(1,1e5,n_tests)
    
    for s in tqdm(seeds):
        algorithm_globals.random_seed = s
        # Run pegasos with statevector as reference
        _, a, _, _ = pegasos(K,y,N,C,seed=s,full_returns=True)

        for i, R in enumerate(shots):
            # Check whether this has already been calculated
            if ((results['seed'] == s) & (results['R'] == R) & (results['C'] == C)).any():
                continue
            
            # Approximate Kernel
            if R == np.inf:
                K_shots = K
            else:
                K_shots = shots_kernel.approximate_kernel(R,seed=s)

            # Run pegasos with finite shots 'R'
            _, a2, _, _ = pegasos(K_shots,y,N,C,seed=s,full_returns=True)
            y2 = np.sign(a2 * y @ K)
            #print(y.shape, y2.shape)
            # Calculating the errors on the weights
            errors_a = np.linalg.norm(a - a2,axis = 1,ord = 1)
            # Calculating accuracy on test and training set
            accuracies = accuracy(np.sign(y2),y)
            accuracies_test = np.zeros_like(accuracies)
            for n in range(N):
                y_n = np.sign((a2[n,:] * y) @ K_test.T)
                accuracies_test[n] = accuracy(y_n,yt)

            # Saving results to csv
            results.loc[results.shape[0]] = [s, R, C, M] + accuracies.tolist() + accuracies_test.tolist() + errors_a.tolist()
            results.to_csv(f'data/pegasos_{margin}_data{N}.csv',index=False)

def create_plots(filename,margin,N,legend=True,shots=None,upto=None):
    """
    Loads the data from 'filename' and creates plots for training and test accuracy, as well as 
    plots showing the evolution of the error on the weights alpha
    """
    # Load data
    data = pd.read_csv(filename)
    if shots is None:
        # Get list of the number of shots used
        shots = list(set(data['R']))
        shots.sort()
    if upto is None:
        upto = N
    # Get list of the regulizers used
    Cs = list(set(data['C']))
    cols = [blue,orange,green,red,violet,cyan]
    # x-axis showing the number of iterations
    x = np.arange(upto)
    # Figure for training accuracy plots
    fig_acc, axs_acc = plt.subplots(len(Cs),sharex=True,figsize=[12,12])
    # Figure for test accuracy plots
    fig_tacc, axs_tacc = plt.subplots(len(Cs),sharex=True,figsize=[12,12])
    # Figure for error evolution plots
    fig_a, axs_a = plt.subplots(len(Cs),sharex=True,figsize=[12,12])

    for j, C in enumerate(Cs):
        for i, R in enumerate(shots):
            acc = np.array(data.loc[(data['R'] == R) & (data['C'] == C)].iloc[:,4:4 + N])
            if acc.shape[0] != 0:
                acc_mean = np.mean(acc,axis=0)
                acc_lower = np.quantile(acc, lower_percentile, axis=0)
                acc_upper = np.quantile(acc, upper_percentile, axis=0)
                
                if R == np.inf:
                    axs_acc[j].plot(x, acc_mean[:upto], label=f'Exact kernel',color=cols[i])
                else:
                    axs_acc[j].plot(x, acc_mean[:upto], label=f'$R={int(R)}$',color=cols[i])
                #axs_acc[j].plot(x, acc_mean[:upto], label=f'$R={int(R)}$')
                axs_acc[j].set_ylim(0.4,1)
                axs_acc[j].fill_between(x, acc_lower[:upto], acc_upper[:upto], alpha=0.3, edgecolor=None,color=cols[i])
                #axs_acc[j].fill_between(x, acc_lower[:upto], acc_upper[:upto], alpha=0.3, edgecolor=None)

            tacc = np.array(data.loc[(data['R'] == R) & (data['C'] == C)].iloc[:,4 + N:4 + 2*N])
            if tacc.shape[0] != 0 and R != np.inf:
                tacc_mean = np.mean(tacc,axis=0)
                tacc_lower = np.quantile(tacc, lower_percentile, axis=0)
                tacc_upper = np.quantile(tacc, upper_percentile, axis=0)
                axs_tacc[j].plot(x, tacc_mean[:upto], label=f'$R={int(R)}$')
                axs_tacc[j].fill_between(x, tacc_lower[:upto], tacc_upper[:upto], alpha=0.3, edgecolor=None)

            a = np.array(data.loc[(data['R'] == R) & (data['C'] == C)].iloc[:,4 + 2*N:])
            if a.shape[0] != 0 and R != np.inf:
                a_mean = np.mean(a,axis=0)
                a_lower = np.quantile(a, lower_percentile, axis=0)
                a_upper = np.quantile(a, upper_percentile, axis=0)
                axs_a[j].plot(x, (a_mean/C)[:upto], label=f'$R={int(R)}$',color=cols[i])
                #axs_a[j].plot(x, (a_mean/C)[:upto], label=f'$R={int(R)}$')
                axs_a[j].fill_between(x, (a_lower/C)[:upto], (a_upper/C)[:upto], alpha=0.3, edgecolor=None,color=cols[i])
                #axs_a[j].fill_between(x, (a_lower/C)[:upto], (a_upper/C)[:upto], alpha=0.3, edgecolor=None)

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

        if legend:
            axs_a[0].legend(loc='upper right')
            axs_tacc[1].legend(loc='lower right')
            axs_acc[1].legend(loc='lower right')
        
    fig_acc.savefig(f'plots/margin{margin}_accuracy.png',dpi=300,bbox_inches='tight')
    fig_a.savefig(f'plots/margin{margin}a_plot.png',dpi=300,bbox_inches='tight')
        

def run_advanced_experiment(margin,C,N,shots,M=1000,M_test=100,n_tests=100):
    # Feature map for the experiment
    feature_map = MediumFeatureMap(2,4)

    # Creating artificial data
    (X,y), (Xt,yt) = generate_qsvm_data(feature_map,margin,M,M_test)

    # Checking whether experiment has already been partially done and loading existing data
    try:
        results = pd.read_csv(f'data/advanced_margin_{margin}_data.csv')
    except:
        columns = ['seed','R','C','M','train acc','epsilon','evaluations']
        results = pd.DataFrame(columns=columns)
        results.to_csv(f'data/advanced_margin_{margin}_data.csv',index=False)

    np.random.seed(41)

    # Change to qasm-simulator if noise is modelled via shots
    adhoc_backend = QuantumInstance(Aer.get_backend('statevector_simulator'))

    adhoc_kernel = QuantumKernel(feature_map=feature_map.get_reduced_params_circuit(), quantum_instance=adhoc_backend)

    # Kernel evaluation
    K = adhoc_kernel.evaluate(x_vec=X)
    K_test = adhoc_kernel.evaluate(x_vec=Xt,y_vec=X)

    # Generating Kernels for different number of shots
    shots_kernel = ShotBasedQuantumKernel(K)
    K_shots = np.zeros((len(shots),) + K.shape)
    

    # Repeating experiment for 100 seeds
    seeds = np.random.randint(1,1e5,n_tests)


    for s in tqdm(seeds):
        algorithm_globals.random_seed = s
        # Running tests
        y_state, a, _, evals = pegasos(K,y,N,C,seed=s,full_returns=True)


        for i, R in enumerate(shots):
            # Check whether this has already been calculated
            if ((results['seed'] == s) & (results['R'] == R) & (results['C'] == C)).any():
                continue
            K_shots = shots_kernel.approximate_kernel(R,seed=s)
            y2,a2,_,evals2 = pegasos(K_shots,y,N,C,seed=s,full_returns=True)
            errors_a = np.linalg.norm(a - a2,axis = 1,ord = 1)
            epsilons = np.array([np.max(np.abs(np.sum(y*(a2[i] - a[-1])*K,axis=1))) for i in range(len(a2))])
            epsilons_2 = np.array([np.max(np.abs(yp - y_state[-1])) for yp in y2])

            accuracies = accuracy(np.sign(y2),y)

            hinges = np.array([hinge_loss(y,y_i) for y_i in y2]) 

            history = {
                'train_accuracy' : accuracies,
                'a_exact' : a,
                'a_noisy' : a2,
                'errors_a' : errors_a,
                'epsilons' : epsilons,
                'epsilons2' : epsilons_2,
                'y_pred' : y2,
                'y_true' : y,
                'y_state' : y_state[-1],
                'evaluations' : evals2,
                'evaluations_exact' : evals,
                'hinge_loss' : hinges
            }

            pickle.dump(history,open(f'data/dumps/{s}_R_{R}_C_{C}_M_{M}_margin_{margin}.pkl','wb'))

            results.loc[results.shape[0]] = [s, R, C, M, accuracies[-1], epsilons[-1], evals2[-1]]
            results.to_csv(f'data/advanced_margin_{margin}_data.csv',index=False)




if __name__ == "__main__":
    shots = [1,2,8,16,32,64,128,256,512,1024]
    shots = [1,8,32,128,1024,np.inf]
    N = 500
    M = 100
    M_test = 20
    n_tests = 100
    
    margins = [0.1,-0.1]
    Cs = [1000.,10.]

    for margin in margins:
        for C in Cs:
            #continue
            run_experiment(margin,C,N,shots,M,M_test,n_tests=n_tests)
        legend = margin < 0
        create_plots(f'data/pegasos_{margin}_data{N}.csv',margin,N,legend,shots)
   