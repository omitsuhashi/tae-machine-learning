import numpy as np

from fw.stage01.variable import Variable
from fw.stage01.function import Function


class Exp(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.input.data
        return np.exp(x) * gy


def exp(x: Variable) -> Variable:
    f = Exp()
    return f(x)
