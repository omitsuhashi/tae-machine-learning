import numpy as np

from fw.variable import Variable
from src.fw.function import Function


class Exp(Function):
    def forward(self, x: np.ndarray):
        return np.exp(x)

    def backward(self, gy: np.ndarray):
        x = self.input.data
        return np.exp(x) * gy


def exp(x: Variable) -> Variable:
    f = Exp()
    return f(x)
