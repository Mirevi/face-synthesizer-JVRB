from abc import ABC, abstractmethod

from torch import nn


class TraceableNetwork(nn.Module, ABC):
    def __init__(self):
        super(TraceableNetwork, self).__init__()

    @abstractmethod
    def input_noise(self, metadata):
        pass
