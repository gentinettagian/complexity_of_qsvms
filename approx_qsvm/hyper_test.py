
# Necessary imports

import numpy as np
import pandas as pd
import pickle

from qiskit  import Aer
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.algorithms.optimizers import SPSA, GradientDescent

from qiskit_machine_learning.utils.loss_functions import CrossEntropyLoss


class HyperParamsTest():
    """
    Running QSVM Tests
    """
    def __init__(self, d=2, seed = 42, reps = 3, initial_weights = None, batch_size = 5, num_steps = 1000, tol = 1e-4) -> None:
        """
        q: number of qubits
        r: feature map repetitions
        d: variational form number of parameters
        seed: random seed used to sample shots and generate data
        reps: repetitions of the variational form
        initial_weights: initial weights for the trainable parameters
        """
        
        # QASM-simulator used for the SPSA optimization
        self._backend = QuantumInstance(Aer.get_backend('statevector_simulator'))
        self.d = d
        self.seed = seed
        self._reps = reps
        self.batch_size = batch_size
        
        # Seed for initial weights should be different from the generated data
        np.random.seed(2*seed)
        if initial_weights is None:
            initial_weights = 0.1*(2*np.random.rand(self.d*(reps + 1)) - 1)
        self._weight = initial_weights
        print(self._weight)

        # variational quantum circuit used to perform optimization
        self._model = VQC(self.d,reps=self._reps,quantum_instance=self._backend,initial_point=self._weight, batch_size=self.batch_size)


        self._x_test = None
        self._y_test = None
        self._x_train = None
        self._y_train = None
        self._true_theta = None

        self._num_steps = num_steps
        self._num_evals = 0

        self._tol = tol
        self._conv_steps = np.zeros(len(tol))
        self._final_loss = np.zeros(len(tol)) 
        self._final_acc = np.zeros(len(tol))
    
        self._loss = CrossEntropyLoss()

        # Dictionary containing data accumulated during training
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
        X,y,_,theta = get_data_generated(self._model.neural_network,margin=margin,M=M+M_test,return_theta=True,seed=seed)
        
        self._x_train = X[:M,:]
        self._y_train = y[:M]
        
        if M_test > 0:
            self._x_test = X[M:,:]
            self._y_test = y[M:]


        self._true_theta = theta
        self.history['theta_true'] = theta
        self._true_h = self._model.neural_network.forward(self._x_train, self._true_theta)
        self.history['h_true'] = self._true_h
        print(X.shape, y.shape)

        return X,y,theta
    
    def fit_model(self):
        """
        Perform SPSA optimization using QASM simulator
        """
        if self._x_train is None:
            RuntimeError('Data not generated')
         
        self._model.fit(self._x_train,self._y_train)
        
        h_fit = self._model.neural_network.forward(self._x_train,self._weight)
 
        return h_fit
    
    
    def run_experiment(self, M = 100, M_test = 10, margin = 0.1):
        """
        Runs the experiment by generating data, fitting with SPSA and 
        controling with gradient descent and statevector
        """

        # Callback used to save data on the fly
        def callback(*args):
            self.history["params"].append(args[1])
            self._weight = args[1]
            self._num_evals = args[0]
            n = len(self.history['loss'])
            
            h_pred = self._model.neural_network.forward(self._x_train, self._weight)
            loss = np.mean(self._loss.evaluate(h_pred, self._y_train))
            self.history["loss"].append(loss)
            y_pred = [[0,1] if p[0] < p[1] else [1,0] for p in h_pred]
            acc = np.sum(y_pred == self._y_train)/(2*len(y_pred))
            self.history["accuracy"].append(acc)
            print(f"{n}, Accuracy: {acc}, Loss: {loss}")

            if n < 2:
                return False
            for i, t in enumerate(self._tol):
                if self._conv_steps[i] == 0 and np.abs(self.history['loss'][-1] - self.history['loss'][-2]) < t:
                    self._conv_steps[i] = n
                    self._final_acc[i] = acc
                    self._final_loss[i] = loss
                    print(f'Tolerance {t} reached.')
                
            if np.all(self._conv_steps > 0):
                return True
            else:
                return False


        
        optimizer = SPSA(maxiter=self._num_steps,termination_checker=callback)
        self._model = VQC(self.d,reps=self._reps,quantum_instance=self._backend,initial_point=self._weight,optimizer=optimizer, batch_size=self.batch_size)

        
        if self._x_train is None:
            self.generate_data(M, M_test, margin,seed=self.seed)
        
        print('Starting qasm fit')
        h_fit = self.fit_model()
        self.history['h'] = h_fit

        return h_fit, self._conv_steps, self._final_loss, self._final_acc
    
    def save(self, filename):
        """
        Saves the history dictionary to a pickle file.
        """
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
    np.random.seed(42)
    seeds = np.random.randint(0,100000,100)
    reps = 3
    features = 2
    margin = 0.1
    sep = 'separable' if margin > 0 else 'overlap'
    try:
        df = pd.read_csv(f'features={features}/d={features*(reps+1)}/hyper_{sep}.csv')
    except:
        df = pd.DataFrame(columns=['Seed','Batch','Tol','Convergence','Loss','Accuracy'])


    n = 1000

    batches = [1,3,5,10,20]
    tol = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]


    for s in seeds:
        for batch in batches:
            print(f'Seed {s}, {batch} samples per batch.')
            if np.any((df['Seed'] == s) & (df['Batch'] == batch)):
                continue
            test = HyperParamsTest(d=features,num_steps=n,seed=s,reps=reps,batch_size=batch, tol=tol)
            h, convergences, losses, accuracies = test.run_experiment(margin=margin)
            test.save(f'{sep}_hyper_seed_{s}_batch_{batch}_steps')

            for i, t in enumerate(tol):
                df = df.append({'Seed':s,'Batch': batch,'Tol': tol, 'Convergence': convergences[i],'Loss': losses[i],'Accuracy': accuracies[i]}, ignore_index=True)

            df.to_csv(f'features={features}/d={features*(reps+1)}/hyper_{sep}.csv',index=False)
   
