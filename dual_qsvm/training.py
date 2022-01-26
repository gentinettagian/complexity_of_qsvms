from multiprocessing import Value
import numpy as np
import matplotlib.pyplot as plt
from qiskit.algorithms import optimizers
from scipy.linalg.special_matrices import hadamard
from seaborn import color_palette
from datetime import date, datetime
import platform
import os
from pathos.multiprocessing import ProcessingPool as Pool

from sklearn.datasets import load_iris, make_blobs, make_circles, make_moons
from sklearn.preprocessing import MinMaxScaler

from qiskit.opflow import CircuitStateFn
from qiskit.algorithms.optimizers import ADAM, COBYLA, SPSA
from modified.my_adam import MyADAM
from modified.my_cobyla import MyCOBYLA
from modified.my_spsa import MySPSA

from surfer.qfi import ReverseQFI

from general_functions import get_date_str, print_to_file, make_meshgrid
from effective_dimension import effective_dimension

from feature_maps import EasyFeatureMap, MediumFeatureMap, HardFeatureMap, HadamardFeatureMap
from variational_forms import StandardVariationalForm
from quantum_neural_networks import QuantumNeuralNetwork, QuantumRepetitiveNetwork

np.random.seed(42)

########################################################################################################################

def cross_entropy_loss(y_pred, y):
    """Cross entropy loss for binary classification.
    :param y_pred: float in [0, 1], the probability assigned to class 1 (class 0 has probability 1 - y_pred)
    :param y: int, either 0 or 1, the ground truth label
    :returns: float in [0, +inf)
    """
    if y == 1:
        return -np.log(y_pred)
    else:
        return -np.log(1 - y_pred)

def predict(qnn, theta, X):
    """
    :param qnn: instance of the QuantumNeuralNetwork class
    :param theta: ndarray of shape (d,), the vf parameters in the qnn 
    :param X: ndarray of shape (x_samples, s), classical input features
    :returns: ndarray of shape (x_samples,) containing the predicted probability of class 1 
    """
    y_pred = np.zeros(X.shape[0])
    # loop over all of the examples in X
    for i, x in enumerate(X):
        # bind the parameters in the feature map
        fm_dict = qnn.get_fm_param_dict(x)
        circuit = qnn.circuit.bind_parameters(fm_dict)

        # bind the parameters in the variational form
        vf_dict = qnn.get_vf_param_dict(theta)
        circuit = circuit.bind_parameters(vf_dict)

        # evaluate the circuit
        sv = qnn.sampler.convert(CircuitStateFn(circuit)).primitive

        # convert the amplitudes to probabilities
        probs = sv.probabilities()

        # probability that it's class 1, given by the sum of all terms of odd Hamming weight 
        probs_odd = probs[np.array([bin(i).count('1') % 2 == 1 for i in range(len(probs))])]
        y_pred_i = np.sum(probs_odd)

        y_pred[i] = y_pred_i

    return y_pred

def classify(qnn, theta, X):
    """
    :param qnn: instance of the QuantumNeuralNetwork class
    :param theta: ndarray of shape (d,), the vf parameters in the qnn 
    :param X: ndarray of shape (x_samples, s), classical input features
    :returns: ndarray of shape (x_samples,) containing the predicted class 
    """
    y_pred = predict(qnn, theta, X)

    y_class = np.zeros(y_pred.shape[0], dtype=int)
    for i, y_pred_i in enumerate(y_pred):
        if 0.5 < y_pred_i:
            y_class[i] = 1
        else:
            y_class[i] = 0

    return y_class

def batch_loss(qnn, theta, X, y, loss_function=cross_entropy_loss):
    """
    :param qnn: instance of the QuantumNeuralNetwork class
    :param theta: ndarray of shape (d,), the vf parameters for which the loss is evaluated/that are trained 
    :param X: ndarray of shape (x_samples, s), classical input features
    :param y: ndarray of shape (s,), labels of the classical input (ground truth)
    :returns: average loss of the batch
    """
    y_pred = predict(qnn, theta, X)
    
    loss = 0

    for i, y_pred_i in enumerate(y_pred):
        loss += loss_function(y_pred_i, y[i])

    loss /= y_pred.shape[0]

    return loss

def generate_folder(NAME):
    # date string for folder name
    date = get_date_str()

    # define path according to system
    if platform.system() == 'Windows':
        OUTPATH = f'C:/Users/ArneThomsen/Desktop/plots/training/{date}/'
    elif platform.system() == 'Linux':
        OUTPATH = f'/home/ubuntu/plots/training/{date}/'
    else:
        raise PermissionError('wrong system, must be either Windows or Linux')

    # create output folder
    if not os.path.exists(OUTPATH):
        os.makedirs(OUTPATH)

    FOLDERPATH = OUTPATH + NAME + '/'
    if not os.path.exists(FOLDERPATH):
        os.makedirs(FOLDERPATH)

    return OUTPATH, FOLDERPATH

def plot_loss(loss_array, grad_array, PLOTPATH):
    """Plots the loss as a function of the training epochs.
    :param loss_array: ndarray of shape (epochs,) or (inits, epochs) containing the loss values for training iterations
    :param grad_array: ndarray of shape (epochs,) or (inits, epochs) containing the gradient norms
    :param PLOTPATH: where the resulting plot is saved
    """
    fig, ax1 = plt.subplots(figsize=(12,8))

    if np.ndim(loss_array) == 1:
        epoch_range = np.arange(1, len(loss_array)+1)

        ax1.plot(epoch_range, loss_array, label='loss', color='tab:blue')

        ax2 = ax1.twinx()
        ax2.plot(epoch_range, grad_array, label='grad', color='tab:orange')

    elif np.ndim(loss_array) == 2:
        epoch_range = np.arange(1, loss_array.shape[1]+1)

        # loss
        loss_mean = np.mean(loss_array, axis=0)
        loss_std = np.std(loss_array, axis=0)

        ax1.plot(epoch_range, loss_mean, color='tab:blue')
        ax1.fill_between(epoch_range, loss_mean-loss_std, loss_mean+loss_std, color='tab:blue', alpha=0.3, edgecolor=None)

        # gradient
        grad_mean = np.mean(grad_array, axis=0)
        grad_std = np.std(grad_array, axis=0)

        ax2 = ax1.twinx()
        ax1.plot(epoch_range, grad_mean, color='tab:orange')
        ax1.fill_between(epoch_range, grad_mean-grad_std, grad_mean+grad_std, color='tab:orange', alpha=0.3, edgecolor=None)

    ax1.set(xlabel='epoch', ylim=(0, 1))
    ax1.set_ylabel('average loss', color='tab:blue')
    ax1.axhline(np.log(2), color='k')
    ax1.grid(True)
    ax2.set(ylim=(0, 1))
    ax2.set_ylabel('gradient', color='tab:orange')

    fig.savefig(PLOTPATH + '_loss.png', dpi=300, bbox_inches='tight')

def plot_decision(qnn, params, X, y, ds_name, PLOTPATH):
    """ when training with hinge loss """
    theta = params[:qnn.d]
    a = params[-2]
    b = params[-1]

    fig, ax = plt.subplots(figsize=(12,12))

    xx, yy = make_meshgrid(X)
    Z = qnn.predict_expect(np.c_[xx.ravel(), yy.ravel()], theta).reshape(xx.shape)

    ax.pcolormesh(xx, yy, Z, cmap=plt.cm.coolwarm)
    ax.contour(xx, yy, Z, levels=[b], color='k')

    # data
    ax.scatter(X[:,0][y==0], X[:,1][y==0], color='b')
    ax.scatter(X[:,0][y==1], X[:,1][y==1], color='r')

    fig.savefig(PLOTPATH + '_decision.png', dpi=300, bbox_inches='tight')

def plot_ed(qnn, theta_array, X, n, PLOTPATH):
    """Plots the normalized effective dimension, evaluated at a single theta instead of the MC integral as a function of 
    the training epochs.
    :param qnn: instance of the QuantumNeuralNetwork class
    :param theta_array: ndarray of shape (epochs, d)
    :param X: classical input to the feature map
    :param n: int, determines the resolution in the effective dimension
    :param PLOTPATH: where the resulting plot is saved
    :returns: an ndarray of shape (epochs,) containing the effective dimension evaluated for the different theta
    """
    rev = ReverseQFI()

    x_samples = X.shape[0]
    # only one fixed theta, the one that was found in the training. No MC integration!
    theta_samples = 1

    ed_array = np.zeros(theta_array.shape[0])
    for i, theta in enumerate(theta_array):

        qfi_array = np.zeros((x_samples, theta_samples, qnn.d, qnn.d))
        for j, x in enumerate(X):
            fm_dict = qnn.feature_map.get_param_dict(x)
            circuit = qnn.circuit.bind_parameters(fm_dict)

            qfi_array[j, 0] = rev.compute(circuit, theta)

        ed_array[i] = effective_dimension(qfi_array, n) / qnn.d

    fig, ax = plt.subplots(figsize=(12,8))

    ax.plot(ed_array)

    title = f'n = {n:.0e}'
    ax.set(xlabel='epoch', ylabel='normalized effective dimension', title=title)
    ax.grid(True)
    fig.savefig(PLOTPATH + '_ed.png', dpi=300, bbox_inches='tight')

    return ed_array

def get_iris():
    """iris dataset as a first toy example"""
    ds_name = 'iris'
    iris = load_iris()
    # only consider the first two classes
    X = iris['data'][:100]
    y = iris['target'][:100]
    # pre process the data
    X = MinMaxScaler().fit_transform(X)

    return X, y, ds_name

def get_blobs():
    """blobs dataset is perfectly linearly separable and even simpler than iris"""
    ds_name = 'blobs'
    X, y = make_blobs(n_samples=100, n_features=4, centers=2, random_state=42)
    X = MinMaxScaler().fit_transform(X)

    return X, y, ds_name

def get_blobs_2d():
    """blobs dataset is perfectly linearly separable and even simpler than iris"""
    ds_name = 'blobs2d'
    X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=3)
    X = MinMaxScaler().fit_transform(X)

    return X, y, ds_name

def get_data_2d(name='blobs'):
    """returns a toy dataset with two features to be used for visualization of classifier performance
    :param name: string, one of 'blobs', 'moons' and 'circles'
    """
    if name == 'blobs':
        X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=3)
    elif name == 'moons':
        X, y = make_moons(n_samples=100, random_state=3, noise=0.1)
    elif name == 'circles':
        X, y = make_circles(n_samples=100, random_state=3, noise=0.1, factor=0.5)
    else:
        raise ValueError

    X = MinMaxScaler().fit_transform(X)

    return X, y, name

########################################################################################################################

def training(suffix=''):
    """Non parallelized"""
    t0 = datetime.now()

    # QNN circuit constants
    s = 2
    q = s
    d_list = [4, 8, 16, 32]

    qnn_list = [
        QuantumNeuralNetwork(EasyFeatureMap(q, 1), StandardVariationalForm(q, d, rotation_blocks=['rx', 'ry'])) for d in d_list
    ]

    # load training data
    data_list = [
        get_data_2d('blobs'),
        # get_data_2d('moons'),
        # get_data_2d('circles')
    ]

    # optimization constants
    epochs = 10

    for qnn in qnn_list:

        for X, y, ds_name in data_list:
            NAME = f'{ds_name}_{qnn.designator}_eps={epochs}'
            NAME += suffix

            OUTPATH, FOLDERPATH = generate_folder(NAME)
            with open(OUTPATH + NAME + '.log', 'w') as f:
                qnn.print_designator(f, FOLDERPATH)

                t0_training = datetime.now()
                # loss_list, grad_list, param_list = qnn.fit_stefan(X, y, f=f, lamda=0.1, lr=0.1, maxiter=epochs)
                loss_list, grad_list, param_list = qnn.fit(X, y, f=f, lr=0.1, maxiter=epochs)

                print_to_file(f'{str(datetime.now() - t0_training)[:-7]}, loss = {loss_list[-1]:.4f}\n', f)
                np.save(FOLDERPATH + 'loss_array.npy', loss_list)   # shape (epochs,)
                np.save(FOLDERPATH + 'grad_array.npy', grad_list)   # shape (epochs,)
                np.save(FOLDERPATH + 'param_array.npy', param_list) # shape (epochs, d)

                # plots
                plot_loss(loss_list, grad_list, OUTPATH + NAME)
                plot_decision(qnn, param_list[-1], X, y, ds_name, OUTPATH + NAME)

    print(f'\nprogram: {str(datetime.now() - t0)[:-7]}')

def training_old(suffix=''):
    """Non parallelized"""
    t0 = datetime.now()

    # load training data
    X, y, ds_name = get_blobs_2d()

    # QNN circuit constants
    s = X.shape[1]
    q = s
    d = 16

    qnn_list = [
        QuantumRepetitiveNetwork(MediumFeatureMap(q, 1, hadamard=False), d, rotation_blocks=['rx', 'ry']),
        QuantumNeuralNetwork(MediumFeatureMap(q, 1), StandardVariationalForm(q, d, rotation_blocks=['rx', 'ry'])),
    ]

    # optimization constants
    epochs = 100
    initial_theta = np.random.uniform(-2*np.pi, 2*np.pi, size=d)

    for qnn in qnn_list:
        # optimize the theta while the other arguments are fixed
        objective_function = lambda theta: batch_loss(qnn, theta, X, y)

        NAME = f'{ds_name}_{qnn.designator}_eps={epochs}'
        NAME += suffix

        OUTPATH, FOLDERPATH = generate_folder(NAME)
        with open(OUTPATH + NAME + '.log', 'w') as f:
            qnn.print_designator(f, FOLDERPATH)

            # ADAM
            t0_training = datetime.now()
            optimizer = MyADAM(maxiter=epochs, lr=0.1, file=f)
            loss_list, grad_list, theta_list = optimizer.optimize(d, objective_function, initial_point=initial_theta)
            print_to_file(f'\nADAM: {str(datetime.now() - t0_training)[:-7]}, loss = {loss_list[-1]:.4f}\n', f)
            np.save(FOLDERPATH + 'ADAM_loss_array.npy', loss_list)   # shape (epochs,)
            np.save(FOLDERPATH + 'ADAM_grad_array.npy', grad_list)   # shape (epochs,)
            np.save(FOLDERPATH + 'ADAM_theta_array.npy', theta_list) # shape (epochs, d)

            # SPSA
            # t0_training = datetime.now()
            # optimizer = MySPSA(maxiter=epochs, file=f)
            # loss_list, grad_list, theta_list = optimizer.optimize(d, objective_function, initial_point=initial_theta)
            # print_to_file(f'\nSPSA: {str(datetime.now() - t0_training)[:-7]}, loss = {loss_list[-1]:.4f}\n', f)
            # np.save(FOLDERPATH + 'SPSA_loss_array.npy', loss_list)   # shape (epochs,)
            # np.save(FOLDERPATH + 'SPSA_grad_array.npy', grad_list)   # shape (epochs,)
            # np.save(FOLDERPATH + 'SPSA_theta_array.npy', theta_list) # shape (epochs, d)

            # COBYLA
            # t0_training = datetime.now()
            # optimizer = MyCOBYLA(maxiter=10*epochs)
            # opt_params, value, _ = optimizer.optimize(d, objective_function, initial_point=initial_theta)
            # print_to_file(f'\nCOBYLA: {str(datetime.now() - t0_training)[:-7]}, loss = {value:.4f}\n', f)

            # plots
            # plot_loss(loss_list, grad_list, OUTPATH + NAME)
            # ed_array = plot_ed(qrn, np.array(theta_list), X, n, OUTPATH + NAME)
            # np.save(FOLDERPATH + 'ed_array.npy', ed_array)      # shape (epochs,)

    print(f'\nprogram: {str(datetime.now() - t0)[:-7]}')

def training_parallel(suffix=''):
    """parallelized in different initializations of the variational form parameters"""
    t0 = datetime.now()

    data_list = [
        get_data_2d('blobs'),
        # get_data_2d('moons'),
        # get_data_2d('circles')
    ]

    # define the QNN circuit
    s = 2
    q = s

    d_list = [4, 8, 16]
    # rot_list = ['ry', ['ry', 'rx'], ['rx', 'ry'], ['ry', 'rz'], ['rz', 'ry']]
    rot_list = [['rx', 'ry']]
    # ent_list = ['cx', 'crx']
    ent_list = ['cx']

    qnn_list = [
        QuantumNeuralNetwork(MediumFeatureMap(q, 2), StandardVariationalForm(q, d, rotation_blocks=rot, entanglement_blocks=ent)) 
        # QuantumRepetitiveNetwork(EasyFeatureMap(q, 1), d, rotation_blocks=rot, entanglement_blocks=ent)
        for d in d_list 
        for rot in rot_list 
        for ent in ent_list
    ]

    # qnn_list = [QuantumNeuralNetwork(EasyFeatureMap(q, 1), StandardVariationalForm(q, 4, rotation_blocks=['ry', 'rx']))]

    # number of train steps (where the loss is calculated over the whole dataset and hence "epoch")
    epochs = 100

    optimizer_list = [
        MyADAM(maxiter=epochs, lr=0.1),
        # MySPSA(maxiter=epochs),
        # MyCOBYLA(maxiter=epochs)
    ]

    ############################################################################

    # number of random initializations of the parameters
    inits = os.cpu_count()
    # used to parallelize
    pool = Pool()

    for qnn in qnn_list:

        for X, y, ds_name in data_list:
            # optimize the theta while the other arguments are fixed
            objective_function = lambda theta: batch_loss(qnn, theta, X, y)

            initial_theta = np.random.uniform(-2*np.pi, 2*np.pi, size=(inits, qnn.d))

            NAME = f'{ds_name}_{qnn.designator}_eps={epochs}'
            NAME += suffix

            OUTPATH, FOLDERPATH = generate_folder(NAME)
            with open(OUTPATH + NAME + '.log', 'w') as f:
                qnn.print_designator(f, FOLDERPATH)

                # start the optimization from inits different starting points
                argument_list = [(initial_theta.shape[1], objective_function, initial_theta[i]) for i in range(inits)]

                for optimizer in optimizer_list:
                    t0_training = datetime.now()
                    result = pool.map(optimizer.optimize_parallel, argument_list)
                    loss_array, grad_array, theta_array = optimizer.save_result_parallel(result, FOLDERPATH)
                    print_to_file(f'\n{optimizer.name}: {str(datetime.now() - t0_training)[:-7]}\n', f)

                    plot_loss(loss_array, grad_array, OUTPATH + NAME)

    print(f'\nprogram: {str(datetime.now() - t0)[:-7]}')

def training_parallel_old(suffix=''):
    """parallelized in different initializations of the variational form parameters"""
    t0 = datetime.now()

    # load training data
    X, y, ds_name = get_blobs()

    # define the QNN circuit
    s = X.shape[1]
    q = s

    qnn_list = [
        QuantumNeuralNetwork(MediumFeatureMap(q, 2), StandardVariationalForm(q, d)) for d in [q, 2*q, 4*q, 8*q]
        # QuantumRepetitiveNetwork(EasyFeatureMap(q, q, hadamard=False), d)
    ]

    # number of train steps (where the loss is calculated over the whole dataset and hence "epoch")
    epochs = 100

    optimizer_list = [
        MyADAM(maxiter=epochs, lr=0.1),
        # MySPSA(maxiter=epochs),
        # MyCOBYLA(maxiter=epochs)
    ]

    ############################################################################

    # number of random initializations of the parameters
    inits = os.cpu_count()
    # used to parallelize
    pool = Pool()

    for qnn in qnn_list:
        # optimize the theta while the other arguments are fixed
        objective_function = lambda theta: batch_loss(qnn, theta, X, y)

        initial_theta = np.random.uniform(-2*np.pi, 2*np.pi, size=(inits, qnn.d))

        NAME = f'{ds_name}_{qnn.designator}_eps={epochs}_parallel'
        NAME += suffix

        OUTPATH, FOLDERPATH = generate_folder(NAME)
        with open(OUTPATH + NAME + '.log', 'w') as f:
            qnn.print_designator(f, FOLDERPATH)

            # start the optimization from inits different starting points
            argument_list = [(initial_theta.shape[1], objective_function, initial_theta[i]) for i in range(inits)]

            for optimizer in optimizer_list:
                print_to_file(f'begin {optimizer.name}', f)
                t0_training = datetime.now()
                result = pool.map(optimizer.optimize_parallel, argument_list)
                optimizer.save_result_parallel(result, FOLDERPATH)
                print_to_file(f'\n{optimizer.name}: {str(datetime.now() - t0_training)[:-7]}\n', f)

    print(f'\nprogram: {str(datetime.now() - t0)[:-7]}')

def tests():
    X, y, _ = get_blobs()

    # define constants
    epochs = 5

    # define the QRN circuit
    s = X.shape[1]
    q = s

    r = 1
    fm = EasyFeatureMap(q, r, hadamard=False)

    d = 3*q
    qnn = QuantumRepetitiveNetwork(fm, d, entanglement='full')

    objective_function = lambda theta: batch_loss(qnn, theta, X, y)
    initial_theta = np.random.uniform(-2*np.pi, 2*np.pi, size=d)

    t0 = datetime.now()
    optimizer = MyADAM(maxiter=epochs, lr=0.1)
    loss_list, grad_list, theta_list = optimizer.optimize(d, objective_function, initial_point=initial_theta)
    print(f'\nOpFlow gradients: {str(datetime.now() - t0)[:-7]}, loss = {loss_list[-1]:.4f}\n')

    print(np.array(loss_list).shape)
    print(np.array(grad_list).shape)
    print(np.array(theta_list).shape)

########################################################################################################################

if __name__ == '__main__':
    training('_test')
    # training_parallel('_log_loss')

    # tests()
