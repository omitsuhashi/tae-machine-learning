import numpy as np

from fw.stage02.variable import Variable
from fw.stage02.function import Function


class Square(Function):
    def forward(self, *xs: np.ndarray) -> tuple[np.ndarray]:
        ys: list[np.ndarray] = [x ** 2 for x in xs]
        return tuple(ys)

    def backward(self, gy: np.ndarray) -> tuple[np.ndarray]:
        xs = [x.data for x in self.inputs]
        return 2 * xs * gy


def square(*xs: Variable) -> tuple[Variable]:
    f = Square()
    return f(*xs)
