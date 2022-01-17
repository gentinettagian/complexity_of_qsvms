import numpy as np
from inspect import signature

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliTwoDesign, TwoLocal

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

########################################################################################################################
# variational forms specified by d

class StandardVariationalForm(_VariationalForm):
    """Own implementation of the TwoLocal circuit that allows to specifiy the number of parameters instead of the number
    of layers/repititions in the circuit. Identical (apart from final entangling layer) to PaperVariationalForm for 
    default parameters."""
    name = 'standard'

    def __init__(self, q, d, rotation_blocks='ry', entanglement_blocks='cx', entanglement='full'):
        """
        :param q: number of qubits
        :param d: number of parameters in the variational form
        :param rotation_blocks: str or list of str, see rot_gate_dict for which are allowed. Determines the kind of
                                gate used in the single qubit gate layer.
        :param entanglement_blocks: str or list of str, see ent_gate_dict for which are allowed. Determines the kind of
                                    gate used in the entanglement layer.
        :param entanglement: str, entanglement strategy. One of 'full', 'linear' and 'circular'
        """
        super().__init__(q, d, entanglement)

        self.rotation_blocks = rotation_blocks
        self.entanglement_blocks = entanglement_blocks

        self.designator_loose = f'vf  = {self.name}\nd   = {self.d}\nrot = {self.rotation_blocks}\nent = {self.entanglement_blocks}, {self.entanglement}'
        self.designator = self.designator_loose.replace(' ', '').replace('\n', ',')
    
        # fill up with parameterized rotations until there are no parameters left
        while True:
            try:
                # make a list out of the single string such that the following for loop always works
                if isinstance(rotation_blocks, str):
                    rotation_blocks = [rotation_blocks]

                elif not isinstance(rotation_blocks, list):
                    raise TypeError(f'Unsupported argument: rotation_blocks = {rotation_blocks}. \
                    It has to be a str or a list of str.')

                # add single qubit gates
                for rotation_gate in rotation_blocks:
                    for j in range(self.q):
                        try:
                            rot_gate_dict[rotation_gate](self.circuit, self.parameters[next(self.indices)], j)
                        except KeyError:
                            raise ValueError(f'Unsupported gate name: rotation_blocks = {rotation_blocks}')

                # add two qubit gates
                self._add_entanglement_layer(entanglement_blocks=entanglement_blocks)              

            except StopIteration:
                # no additional entangling layer, the circuit ends in single qubit gates like in qiskit TwoLocal                
                break

class ConvolutionalVariationalForm(_VariationalForm):
    """Inspired by the convolutional layers (no 'pooling') in 'Quantum Convolutional Neural Networks' by Cong et al.
    https://arxiv.org/abs/1810.03787. Consists solely of two qubit gates between nearest neighbors."""
    name = 'convolutional'

    def __init__(self, q, d, rotation_blocks=['cry', 'crx'], entanglement_blocks='cx', entanglement=None):
        """
        :param q: number of qubits
        :param d: number of parameters in the variational form
        :param rotation_blocks: list containing two str, see ent_gate_dict for which are allowed. Determines the kind of
                                gate used in the alternating nearest neighbor two qubit gates.
        :param entanglement_blocks: str or list of str, see ent_gate_dict for which are allowed. Determines the kind of
                                    gate used in the entanglement layer.
        :param entanglement: str, entanglement strategy. One of 'full', 'linear' and 'circular'
        """
        super().__init__(q, d, entanglement=entanglement)

        assert isinstance(rotation_blocks, list) and len(rotation_blocks)==2, 'rotation_blocks has to be a list of two str'

        # fill up with parameterized rotations until there are no parameters left
        while True:
            try:
                for i in range(self.q):
                    if i % 2 == 0 and i < q-1:
                        gate_key = rotation_blocks[0]
                        ent_gate_dict[gate_key](self.circuit, self.parameters[next(self.indices)], i, i+1)

                for i in range(self.q):
                    if i % 2 == 1 and i < q-1:
                        gate_key = rotation_blocks[1]
                        ent_gate_dict[gate_key](self.circuit, self.parameters[next(self.indices)], i, i+1)

                self._add_entanglement_layer(entanglement_blocks=entanglement_blocks)

            except StopIteration:
                # no additional entangling layer, the circuit ends in single qubit gates like in qiskit TwoLocal
                break

############################################################################
# legacy

class PaperVariationalForm(_VariationalForm):
    """Variational form consisting of Y rotations and entangling layers. Similar to the variational form in 
    'The Power of QNN' paper, but with a number of parameters that is independent from the number of qubits.
    https://arxiv.org/abs/2011.00027"""
    name = 'paper'

    def __init__(self, q, d, entanglement='full'):
        """
        :param q: number of qubits
        :param d: number of parameters in the variational form
        :param entanglement: str, entanglement strategy. One of 'full', 'linear' and 'circular'
        """
        super().__init__(q, d, entanglement)

        # fill up with parameterized rotations until there are no parameters left
        while True:
            try:
                # single qubit y rotations
                for j in range(self.q):
                    self.circuit.ry(self.parameters[next(self.indices)], j)
                
                self._add_entanglement_layer()              

            except StopIteration:
                # if there is a partially filled final layer of rotations, add another entanglement layer behind
                if self.d % self.q != 0:
                    self._add_entanglement_layer()
                
                break

class rzryVariationalForm(_VariationalForm):
    """Variational form with both y and z rotations and a variable entanglement strategy."""
    name = 'rzry'

    def __init__(self, q, d, entanglement='full'):
        """
        :param q: number of qubits
        :param d: number of parameters in the variational form
        :param entanglement: str, entanglement strategy. One of 'full', 'linear' and 'circular'
        """
        super().__init__(q, d, entanglement)

        # fill up with parameterized rotations until there are no parameters left
        while True:
            try:
                # single qubit y rotations
                for j in range(self.q):
                    self.circuit.ry(self.parameters[next(self.indices)], j)

                # single qubit z rotations
                for j in range(self.q):
                    self.circuit.rz(self.parameters[next(self.indices)], j)

                self._add_entanglement_layer()              

            except StopIteration:
                # if there is a partially filled final layer of rotations, add another entanglement layer behind
                if self.d % self.q != 0:
                    self._add_entanglement_layer()
                
                break

class AnalyticVariationalForm(_VariationalForm):
    """Variational form where the QFI can be calculated analytically."""
    name = 'analytic'

    def __init__(self, q, d, entanglement='full'):
        """
        :param q: number of qubits
        :param d: number of parameters in the variational form
        """
        super().__init__(q, d)
        self.d = d
        # this is just a placeholder and the entire variational form with its random H existis inside the method
        # _evaluate_qfi_analytic of the QuantumNeuralNetwork class

        # TODO empty circuit (without parameters) sets qnn.d = 0 in QuantumNeuralNetwork

class RandomVariationalForm(_VariationalForm):
    """Variational form consisting of randomly placed qubit gates."""
    name = 'random'

    def __init__(self, q, d, entanglement=None, gate='cry'):
        """
        :param q: number of qubits
        :param d: number of parameters in the variational form
        :param entanglement: str, entanglement strategy. One of 'full', 'linear' and 'circular'
        :param gate: str, the gate type the random circuit is constructed from. One of 'cry' and 'ry'
        """
        super().__init__(q, d, entanglement)
        self.gate = gate

        # create d new parameters, starting from zero
        indices = iter(range(self.d))

        self._add_random_circuit(indices)

    def extend(self, d):
        """Extend the existing circuit by subsequent random gates, preserving the previous architecture.
        :param d: number of parameters in the variational form. Has to be larger than current value
        """
        if d > self.d:
            # extend the existing parameter vector to the new larger size
            self.parameters.resize(d)

            # leave the previous indices unchanged
            indices = iter(range(self.d, d))

            # update total parameter number
            self.d = d

            self._add_random_circuit(indices)

        else:
            raise ValueError(f'new d = {d} must be larger than previous d = {self.d}')

    def _add_random_circuit(self, indices):
        """Adds random gates to the existing circuit.
        :param indices: the indices of the parameters that are added
        """
        # fill up with parameterized rotations until there are no parameters left
        while True:
            try:
                i = np.random.randint(self.q)
                j = np.random.randint(self.q)

                index = next(indices)

                # use single qubit rotations
                if self.gate == 'ry':
                    self.circuit.ry(self.parameters[index], i)

                # use entangling controlled y rotations
                elif self.gate == 'cry':

                    if i == j:
                        self.circuit.ry(self.parameters[index], i)

                    else:
                        self.circuit.cry(self.parameters[index], i, j)

                # use entangling controlled two qubit y rotations
                elif self.gate == 'ryy':

                    if i == j:
                        self.circuit.ry(self.parameters[index], i)

                    else:
                        self.circuit.ryy(self.parameters[index], i, j)

                # use entangling controlled x and y rotations
                elif self.gate == 'crxcry':

                    if i == j:
                        if np.random.randint(2) == 0:
                            self.circuit.ry(self.parameters[index], i)
                        else:
                            self.circuit.rx(self.parameters[index], i)

                    else:
                        if np.random.randint(2) == 0:
                            self.circuit.cry(self.parameters[index], i, j)
                        else:
                            self.circuit.crx(self.parameters[index], i, j)

                # use entangling two qubit x and y rotations
                elif self.gate == 'rxxryy':

                    if i == j:
                        if np.random.randint(2) == 0:
                            self.circuit.ry(self.parameters[index], i)
                        else:
                            self.circuit.rx(self.parameters[index], i)

                    else:
                        if np.random.randint(2) == 0:
                            self.circuit.ryy(self.parameters[index], i, j)
                        else:
                            self.circuit.rxx(self.parameters[index], i, j)

                else:
                    raise ValueError(f'Unsupported gate type: {self.gate}')

                # add an entanglement layer after every q gates
                if index % self.q == self.q-1 and index > 0 and self.entanglement is not None:
                    self._add_entanglement_layer(self.entanglement)
                    # self.circuit.barrier()

            except StopIteration:
                break

############################################################################
# deep single layer

class AutoencoderVariationalForm(_VariationalForm):
    """Inspired by the QNN in 'Quantum autoencoders for efficient compression of quantum data' by Romero et al.
    https://arxiv.org/pdf/1612.02806 Fig. 3b"""
    name = 'autoencoder'

    def __init__(self, q, d, entanglement=None):
        """
        :param q: number of qubits
        :param d: number of parameters in the variational form
        :param entanglement: str, entanglement strategy. One of 'full', 'linear' and 'circular'
        """
        super().__init__(q, d, entanglement)

        indices = iter(range(self.d))

        # fill up with parameterized rotations until there are no parameters left
        while True:
            try:
                for i in range(self.q):
                    self.circuit.ry(self.parameters[next(indices)], i)

                for i in range(self.q):
                    for j in range(self.q):
                        if i != j:
                            self.circuit.crx(self.parameters[next(indices)], i, j)

                for i in range(self.q):
                    self.circuit.rz(self.parameters[next(indices)], i)

                self._add_entanglement_layer()              

            except StopIteration:
                # if there is a partially filled final layer of rotations, add another entanglement layer behind
                if self.d % self.q != 0:
                    self._add_entanglement_layer()
                
                break

class FourteenVariationalForm(_VariationalForm):
    """Variational form inspired by Circuit 14 in Fig. 2 of the paper 'Expressibility and entangling capability of
    parametrized quantum circuits for hybrid quantum-classical algorithms' by Sim, et. al."""
    name = 'Circuit_14'

    def __init__(self, q, d):
        """
        :param q: number of qubits
        :param d: number of parameters in the variational form
        """
        super().__init__(q, d)

        thetas = ParameterVector('θ', self.d)
        theta_index = iter(range(len(thetas)))

        # fill up with parameterized rotations until there are no parameters left
        while True:
            try:
                # single qubit y rotations
                for j in range(q):
                    self.circuit.ry(thetas[next(theta_index)], j)

                # controlled single qubit x rotations
                self.circuit.crx(thetas[next(theta_index)], self.q-1, 0)
                for j in reversed(range(self.q)):
                    if j < q-1:
                        self.circuit.crx(thetas[next(theta_index)], j, j+1)

                # self.circuit.barrier()

                # single qubit y rotations
                for j in range(self.q):
                    self.circuit.ry(thetas[next(theta_index)], j)

                # controlled single qubit x rotations
                self.circuit.crx(thetas[next(theta_index)], 0, q-1)
                for j in range(self.q):
                    if j < self.q-1:
                        self.circuit.crx(thetas[next(theta_index)], j+1, j)

                # self.circuit.barrier()

            except StopIteration:
                break

class FourVariationalForm(_VariationalForm):
    """Variational form inspired by Circuit 14 in Fig. 2 of the paper 'Expressibility and entangling capability of
    parametrized quantum circuits for hybrid quantum-classical algorithms' by Sim, et. al."""
    name = 'Circuit_4'

    def __init__(self, q, d):
        """
        :param q: number of qubits
        :param d: number of parameters in the variational form
        """
        super().__init__(q, d)

        thetas = ParameterVector('θ', self.d)
        theta_index = iter(range(len(thetas)))

        # fill up with parameterized rotations until there are no parameters left
        while True:
            try:
                # single qubit y rotations
                for j in range(q):
                    self.circuit.ry(thetas[next(theta_index)], j)

                # single qubit z rotations
                for j in range(q):
                    self.circuit.rz(thetas[next(theta_index)], j)

                # controlled single qubit y rotations between nearest neighbors
                for j in reversed(range(self.q)):
                    if j < self.q-1:
                        self.circuit.cry(thetas[next(theta_index)], j+1, j)

                # self.circuit.barrier()

            except StopIteration:
                break

########################################################################################################################
# variational forms specified by r

class MyPauliTwoDesign(_VariationalForm):
    """Wrapper for qiskit implemented PauliTwoDesign
    https://qiskit.org/documentation/stubs/qiskit.circuit.library.PauliTwoDesign.html#qiskit.circuit.library.PauliTwoDesign"""
    name = 'pauli_two_design'

    def __init__(self, q, r):
        """
        :param q: number of qubits
        :param d: number of parameters in the variational form
        :param entanglement: str, entanglement strategy. One of 'full', 'linear' and 'circular'
        """
        self.q = q
        self.r = r
        self.circuit = PauliTwoDesign(num_qubits=self.q, reps=self.r)
        self.d = self.circuit.num_parameters

class MyTwoLocal(_VariationalForm):
    """Wrapper for qiskit implemented TwoLocal
    https://qiskit.org/documentation/stubs/qiskit.circuit.library.TwoLocal.html#qiskit.circuit.library.TwoLocal"""
    name = 'two_local'

    def __init__(self, q, r, rotation_blocks='ry',  entanglement_blocks='cx', entanglement='linear'):
        """
        :param q: number of qubits
        :param r: number of repetitions in the variational form
        :param entanglement: str, entanglement strategy. One of 'full', 'linear', 'circular' and 'sca'
        """
        self.q = q
        self.r = r
        self.circuit = TwoLocal(num_qubits=self.q, reps=self.r, rotation_blocks=rotation_blocks,
                                entanglement_blocks=entanglement_blocks, entanglement=entanglement)
        self.d = self.circuit.num_parameters
