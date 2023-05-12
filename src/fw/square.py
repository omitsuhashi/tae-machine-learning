import numpy as np

from fw.variable import Variable
from src.fw.function import Function


class Square(Function):
    def forward(self, x: np.ndarray):
        return x ** 2

    def backward(self, gy: np.ndarray):
        x = self.input.data
        return 2 * x * gy


def square(x: Variable) -> Variable:
    f = Square()
    return f(x)
