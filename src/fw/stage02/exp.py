import numpy as np

from fw.stage02.variable import Variable
from fw.stage02.function import Function


class Exp(Function):
    def forward(self, *xs: np.ndarray) -> tuple[np.ndarray]:
        return np.exp(xs)

    def backward(self, *gys: np.ndarray) -> tuple[np.ndarray]:
        xs = [val.data for val in self.inputs]
        return np.exp(xs) * gys


def exp(*xs: Variable) -> tuple[Variable]:
    f = Exp()
    return f(*xs)
