
# Necessary imports

import numpy as np
import matplotlib.pyplot as plt
from qiskit.algorithms import optimizers


#from torch import Tensor
#from torch.nn import Linear, CrossEntropyLoss, MSELoss
#from torch.optim import LBFGS, SGD, Adam

from qiskit  import Aer
from qiskit.utils import QuantumInstance
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.neural_networks import CircuitQNN, TwoLayerQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
from qiskit.algorithms.optimizers import SPSA, GradientDescent, COBYLA

import pandas as pd
import pickle
class ApproxQSVMTest():
    """
    Running QSVM Tests
    """
    def __init__(self, d=2, Rshots = 1024, num_steps = 100, control_steps = 10, seed = 42, reps = 3, initial_weights = None) -> None:
        """
        q: number of qubits
        r: feature map repetitions
        d: variational form number of parameters
        """
        

        self._backend = QuantumInstance(Aer.get_backend('qasm_simulator'),shots=Rshots)
        self.d = d
    
        self.seed = seed
        self._reps = reps
        np.random.seed(2*seed)
        if initial_weights is None:
            initial_weights = 0.1*(2*np.random.rand(self.d*(reps + 1)) - 1)

        self._weight = initial_weights
        self._model = VQC(self.d,reps=self._reps,quantum_instance=self._backend,initial_point=self._weight)

        self._sv_instance = Aer.get_backend('statevector_simulator')
        self._model_sv = VQC(self.d,reps=self._reps,quantum_instance=self._sv_instance,initial_point=self._weight)

        self._x_test = None
        self._y_test = None
        self._x_train = None
        self._y_train = None
        self._true_theta = None

        self._num_steps = num_steps
        self._control_steps = control_steps
        self._num_evals = 0

        self.history = {'accuracy' : [],
                        'loss' : [],
                        'accuracy_control' : [],
                        'loss_control' : [],
                        'params' : [],
                        'params_control' : [],
                        'h' : [],
                        'h_sv' : [],
                        'theta_true' : [],
                        'h_true' : []
                        }

    
    def generate_data(self, M = 100, M_test = 10, margin = 0.1, seed=41):
        """
        Generate artificial data
        """
        X,y,_,theta = get_data_generated(self._model_sv.neural_network,margin=margin,M=M+M_test,return_theta=True,seed=seed)
        
        self._x_train = X[:M,:]
        self._y_train = y[:M]
        
        if M_test > 0:
            self._x_test = X[M:,:]
            self._y_test = y[M:]


        self._true_theta = theta
        self.history['theta_true'] = theta
        self._true_h = self._model_sv.neural_network.forward(self._x_train, self._true_theta)
        self.history['h_true'] = self._true_h
        print(X.shape, y.shape)

        return X,y,theta
    
    def fit_model(self):

        if self._x_train is None:
            RuntimeError('Data not generated')
        
        
        self._model.fit(self._x_train,self._y_train)
        
        h_fit = self._model.neural_network.forward(self._x_train,self._weight)

        
        return h_fit
    
    def fit_statevector(self):
        if self._x_train is None:
            RuntimeError('Data not generated')

        
        self._model_sv.fit(self._x_train,self._y_train)
        
        h_sv = self._model_sv.neural_network.forward(self._x_train,self._weight)
        
        return h_sv
    
    def run_experiment(self, M = 100, M_test = 10, margin = 0.1):

        def callback(*args):
            self.history["loss"].append(args[2])
            self.history["params"].append(args[1])
            self._weight = args[1]
            self._num_evals = args[0]
            print(len(self.history['loss']),args[2])
        
        optimizer = SPSA(maxiter=self._num_steps,callback=callback)
        self._model = VQC(self.d,reps=self._reps,quantum_instance=self._backend,initial_point=self._weight,optimizer=optimizer)

        

        if self._x_train is None:
            self.generate_data(M, M_test, margin,seed=self.seed)
        
        print('Starting qasm fit')
        h_fit = self.fit_model()

        
        
        def callback_sv(*args):
            self.history["loss_control"].append(args[2])
            self.history["params_control"].append(args[1])
            self._weight = args[1]
            print(len(self.history['loss_control']),args[2])

        optimizer_sv = GradientDescent(maxiter=self._control_steps,callback=callback_sv,learning_rate=0.001,perturbation=0.001)
        self._model_sv = VQC(self.d,reps=self._reps,quantum_instance=self._sv_instance,initial_point=self._weight, optimizer=optimizer_sv)

        

        print('Controlling with statevector')
        h_sv = self.fit_statevector()
        self.history['h_sv'] = h_sv
        self.history['h'] = h_fit

        #h_sv = self._true_h
        #c_steps = 10
           
        num_evals = self._num_evals
        #c_steps = len(self.history['loss_control'])
        c_steps = self._control_steps

        return h_fit, h_sv, num_evals, c_steps
    
    def save(self, filename):
        f = open(f'features={self.d}/d={int(self.d * (self._reps+1))}/dumps/{filename}.pkl','wb')
        pickle.dump(self.history,f)




def get_data_generated(qnn, M=100, margin=0.1, bias=0, shuffle=True, seed=41, return_theta=False,one_hot=True):
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
    theta = rng.uniform(0, 2*np.pi, size=len(qnn.weight_params))

    class_0 = []
    class_1 = []

    # loop until the two lists both contain M/2 elements
    while len(class_0) < M//2 or len(class_1) < M//2:
        # generate a random point
        x = rng.uniform(0, 1, size=len(qnn.input_params))
        y_prob = qnn.forward(np.array([x]),theta).flatten()

        # strict class membership criteria if margin > 0
        criterion_0 = y_prob[0] < y_prob[1] - margin/2 + bias
        criterion_1 = y_prob[1] < y_prob[0] - margin/2 + bias

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
    y = np.zeros(M, dtype=int) - 1
    y[M//2:] = 1
    X = np.array(class_0 + class_1)

    if shuffle:
        inds = rng.choice(M, M, replace=False)
        X = X[inds]
        y = y[inds]
    
    if one_hot:
        y_one_hot = np.array([[1 if yi == -1 else 0, 1 if yi == 1 else 0] for yi in y])
        y = y_one_hot


    if return_theta:
        return X, y, 'generated', theta
    else:
        return X, y, 'generated'


if __name__ == '__main__':
    seeds = np.random.randint(0,100000,100)
    reps = 1
    features = 8
    margin = -0.1
    try:
        df = pd.read_csv(f'features={features}/d={features*(reps+1)}/overlap_spsaapprox_tests.csv')
    except:
        df = pd.DataFrame(columns=['Seed','Shots','Evaluations','CSteps','Epsilon'])

    print('updated files')
    Rs = 2**np.arange(1,14,2)
    n = 1000
    c = 30

    for s in seeds:
        for R in Rs:
            print(f'Seed {s}, {R} shots.')
            test = ApproxQSVMTest(Rshots=R,d=features,num_steps=n,seed=s,reps=reps,control_steps=c)
            h, h_sv, n_evals, c_effective = test.run_experiment(margin=margin)
            test.save(f'overlap_spsaseed_{s}_R_{R}_steps_{n_evals}_csteps_{c_effective}')
            eps = np.max(np.abs(h-h_sv))
            df = df.append({'Seed':s,'Shots':R,'Evaluations':n_evals,'CSteps':c,'Epsilon':eps}, ignore_index=True)
            df.to_csv(f'features={features}/d={features*(reps+1)}/overlap_spsaapprox_tests.csv',index=False)
   
