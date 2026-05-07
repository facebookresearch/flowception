import torch
from abc import ABC, abstractmethod

class Scheduler(ABC):
    @abstractmethod
    def get_kappa(self, t):
        pass

    @abstractmethod
    def get_dkappa(self, t):
        pass
    
    def get_kappa(self, t):
        return self.get_dkappa(t) / (1 - self.get_kappa(t))
    
class LinearScheduler(Scheduler):
    def get_kappa(self, t):
        return t

    def get_dkappa(self, t):
        return torch.ones_like(t)
    
class CubicScheduler(Scheduler):
    def get_kappa(self, t):
        return t**3
    
    def get_dkappa(self, t):
        return 3 * t**2
    
def get_kappa_scheduler(name):
    assert name in ["linear", "cubic"]
    if name=="linear":
        return LinearScheduler()
    elif name=="cubic":
        return CubicScheduler