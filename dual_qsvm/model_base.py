from abc import ABC, abstractmethod

class Model(ABC):
    """Abstract base class for feature maps & variational forms"""
    name = None

    @abstractmethod
    def __init__(self):
        raise NotImplementedError()

    @abstractmethod
    def get_param_dict(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_bound_circuit(self, *args, **kwargs):
        raise NotImplementedError()
