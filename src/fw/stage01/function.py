import numpy as np

from fw.stage01.variable import Variable


class Function:
    input: Variable
    output: Variable

    def __call__(self, arg: Variable) -> Variable:
        x = arg.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)
        self.input = arg
        self.output = output
        return output

    def forward(self, x: np.ndarray):
        raise NotImplementedError()

    def backward(self, gy: np.ndarray):
        raise NotImplementedError()
