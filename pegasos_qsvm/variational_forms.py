import numpy as np
from inspect import signature

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from model_base import Model

# dictionaries where the keys are strings that can be used as function arguments and the entries are methods of the
# QuantumCircuit class. The distinction is made between single (rotation) and two qubit (entangling) gates.

# single qubit gates
rot_gate_dict = {
    # rotations
    'rx' : QuantumCircuit.rx,
    'ry' : QuantumCircuit.ry,
    'rz' : QuantumCircuit.rz
}

# two qubit gates
ent_gate_dict = {
    # x axis
    'cx' : QuantumCircuit.cx,
    'crx' : QuantumCircuit.crx,
    'rxx' : QuantumCircuit.rxx,
    # y axis
    'cy' : QuantumCircuit.cy,
    'cry' : QuantumCircuit.cry,
    'ryy' : QuantumCircuit.ryy,
    # z axis
    'cz' : QuantumCircuit.cz,
    'crz' : QuantumCircuit.crz,
    'rzz' : QuantumCircuit.rzz
}

class _VariationalForm(Model):
    """Base class for variational circuits."""
    name = 'VariationalForm'

    def __init__(self, q, d, entanglement='full'):
        """
        :param q: number of qubits
        :param d: number of parameters
        :param entanglement: str, entanglement strategy. One of 'full', 'linear' or 'circular'. No entanglement if None
        """
        self.q = q
        self.d = d

        self.circuit = QuantumCircuit(q)
        self.entanglement = entanglement

        self.parameters = ParameterVector('θ', self.d)
        self.indices = iter(range(self.d))

        self.designator = ''

    def get_param_dict(self, theta):
        """Returns a dictionary mapping the variational form's parameters to randomly sampled values.
        :param theta: ndarray, length d. Vector containing the values to be associated with the circuit parameters
        :return: dict, containing the Parameters as keys and the elements of theta as values
        """
        # create dictionary setting parameters in the feature map to a value corresponding to the random samples
        vf_param = list(self.circuit.parameters)
        zip_it = zip(vf_param, theta)
        param_dict = dict(zip_it)

        return param_dict

    def get_bound_circuit(self, theta):
        """Get a QuantumCircuit where the free parameters in self.circuit are bound to random values.
        :param theta: ndarray, length d. Vector containing the values to be associated with the circuit parameters
        :return: QuantumCircuit, with no free parameters
        """
        param_dict = self.get_param_dict(theta)

        return self.circuit.bind_parameters(param_dict)

    def _add_entanglement_layer(self, entanglement_blocks='cx'):
        """Add a single entangling layer according to the entanglement strategy to the circuit.
        :param entanglement_blocks: str or list of str, containing the two qubits gates to be included in the 
                                    entanglement layer
        """
        # make a list out of the single string such that the following for loop always works
        if isinstance(entanglement_blocks, str):
            entanglement_blocks = [entanglement_blocks]

        elif not isinstance(entanglement_blocks, list):
            raise TypeError(f'Unsupported argument: entanglement_blocks = {entanglement_blocks}. \
            It has to be a str or a list of str.')

        # loop over the gates in block
        for entanglement_gate in entanglement_blocks:
            # make disctinction between gates that take a rotation parameter and those that don't
            gate_method = ent_gate_dict[entanglement_gate]

            # all to all connectivity
            if self.entanglement == 'full':
                for j in range(1,self.q):
                    for k in range(j):
                        if j < self.q and k < self.q-1:
                            self._add_entanglement_gate(gate_method, k, j)

                    # self.circuit.barrier()

            # nearest neighbor connectivity
            if self.entanglement == 'linear' or self.entanglement == 'circular':
                for j in range(self.q):
                    if j < self.q-1:
                        self._add_entanglement_gate(gate_method, j, j+1)

                # self.circuit.barrier()
            
            # additional single connection between furthest apart
            if self.entanglement == 'circular' and self.q > 1:
                self._add_entanglement_gate(gate_method, self.q-1, 0)

                # self.circuit.barrier()

            if self.entanglement is None:
                pass

            if type(self.entanglement) == str and self.entanglement not in ['full', 'linear', 'circular']:
                raise ValueError(f'Unsupported entanglement type: {self.entanglement}')

    def _add_entanglement_gate(self, gate_method, qubit_1, qubit_2):
        """Add a two qubit entangling gate to the circuit.
        :param gate_method: method of the QuantumCircuit class adding a single gate to the circuit
        :param qubit_1: the control qubit
        :param qubit_2: the target qubit
        """
        sig = signature(gate_method)

        # parametrized gate like a controlled single qubit rotation
        if 'theta' in sig.parameters:
            try:
                gate_method(self.circuit, self.parameters[next(self.indices)], qubit_1, qubit_2)
            except KeyError:
                raise ValueError(f'Unsupported two qubit rotation gate')

        # non-parametrized gate like CNOT
        else:
            try:
                gate_method(self.circuit, qubit_1, qubit_2)
            except KeyError:
                raise ValueError(f'Unsupported unparametrized two qubit gate')

