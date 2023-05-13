import numpy as np

from fw.stage02.variable import Variable


class Function:
    inputs: tuple[Variable]
    outputs: tuple[Variable]

    def __call__(self, *args: Variable) -> tuple[Variable]:
        xs = [arg.data for arg in args]
        ys = self.forward(*xs)
        outputs = tuple([Variable(y) for y in ys])
        [output.set_creator(self) for output in outputs]
        self.inputs = args
        self.outputs = outputs
        return outputs

    def forward(self, *xs: np.ndarray) -> tuple[np.ndarray]:
        raise NotImplementedError()

    def backward(self, gy: np.ndarray) -> tuple[np.ndarray]:
        raise NotImplementedError()
