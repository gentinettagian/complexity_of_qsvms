import numpy as np

from qiskit import Aer
from qiskit.opflow import CircuitStateFn, CircuitSampler





class QuantumNeuralNetwork:
    """Extremely minimal version of QNN, used to create artificial data."""
    name = 'qnn'

    def __init__(self, feature_map, variational_form):
        """
        :param feature_map: instance of the _FeatureMap class
        :param variational_form: instance of the _VariationalForm class
        """
        self.feature_map = feature_map
        self.variational_form = variational_form
        self.circuit = feature_map.circuit.compose(variational_form.circuit)
        
        self.d = variational_form.d
        self.q = variational_form.q

        self.designator = f'qnn_[q={self.q};{self.feature_map.designator};{self.variational_form.designator}]'

        # for circuit evaluations
        self.sampler = CircuitSampler(Aer.get_backend('statevector_simulator'))

        # when used as a classifier
        self.theta = None


    def get_fm_param_dict(self, x):
        """Get a dictionary associating the free parameters in the feature map with a vector of classical input.
        :param x: ndarray, shape (s,) containing a single input datum
        :return: dict, containing qiskit Parameter objects as the keys and the components of the datum as the values 
        """
        return self.feature_map.get_param_dict(x)

    def get_vf_param_dict(self, theta):
        """Returns a dictionary mapping the variational form's parameters to theta. Like in the _VariationalForm class.
        :param theta: ndarray, shape (d,) containing the values to be associated with the vf parameters
        :return: dict, containing the Parameters as keys and the elements of theta as values
        """
        return self.variational_form.get_param_dict(theta)

    # binary classification

    def fit_theta(self, theta):
        self.theta = theta

    def predict_proba(self, X, theta=None):
        """
        :param X: ndarray of shape (x_samples, s), classical input features
        :returns: ndarray of shape (x_samples,) containing the predicted probability of class 1 
        """
        # when working with a previously fit model (unlike in training where theta is passed as an argument)
        if theta is None and self.theta is not None:
            theta = self.theta

        assert isinstance(theta, np.ndarray) and theta.size == self.d, f'a theta != {theta} has to be fit or provided'

        y_prob = np.zeros(X.shape[0])
        # loop over all of the examples in X
        for i, x in enumerate(X):
            # bind the parameters in the feature map
            fm_dict = self.get_fm_param_dict(x)
            circuit = self.circuit.bind_parameters(fm_dict)

            # bind the parameters in the variational form
            vf_dict = self.get_vf_param_dict(theta)
            circuit = circuit.bind_parameters(vf_dict)

            # evaluate the circuit
            sv = self.sampler.convert(CircuitStateFn(circuit)).primitive

            # convert the amplitudes to probabilities
            probs = sv.probabilities()

            # probability that it's class 1, given by the sum of all terms of odd Hamming weight 
            probs_odd = probs[np.array([bin(i).count('1') % 2 == 1 for i in range(len(probs))])]
            y_prob_i = np.sum(probs_odd)

            y_prob[i] = y_prob_i

        return y_prob

