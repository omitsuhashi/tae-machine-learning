from __future__ import annotations
from typing import Optional, TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    from fw.stage01.function import Function


class Variable:
    __data: np.ndarray

    grad: Optional[np.ndarray] = None
    creator: Optional[Function] = None

    @property
    def data(self) -> np.ndarray:
        return self.__data

    @data.setter
    def data(self, value: Union[np.ndarray, np.number]):
        if np.isscalar(value):
            self.__data = np.array(value)
        self.__data = value

    def __init__(self, data: np.ndarray):
        self.__data = data

    def set_creator(self, func: Function):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)
