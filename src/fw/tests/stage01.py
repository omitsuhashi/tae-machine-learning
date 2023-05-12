import unittest

import numpy as np

from src.fw.exp import Exp, exp
from src.fw.square import Square, square
from src.fw.variable import Variable


class Stage01TestCase(unittest.TestCase):
    def test_step03(self):
        sq_1 = Square()
        e = Exp()
        sq_2 = Square()

        x = Variable(np.array(0.5))
        a = sq_1(x)
        b = e(a)
        y = sq_2(b)
        self.assertEqual(y.data, 1.648721270700128)

    def test_step06(self):
        sq_1 = Square()
        e = Exp()
        sq_2 = Square()

        x = Variable(np.array(0.5))
        a = sq_1(x)
        b = e(a)
        y = sq_2(b)

        y.grad = np.array(1.0)
        b.grad = sq_2.backward(y.grad)
        a.grad = e.backward(b.grad)
        x.grad = sq_1.backward(a.grad)
        self.assertEqual(x.grad, 3.297442541400256)

    def test_step07(self):
        sq_1 = Square()
        e = Exp()
        sq_2 = Square()

        x = Variable(np.array(0.5))
        a = sq_1(x)
        b = e(a)
        y = sq_2(b)

        # ノードをたどる
        self.assertEqual(y.creator, sq_2)
        self.assertEqual(y.creator.input, b)
        self.assertEqual(y.creator.input.creator, e)
        self.assertEqual(y.creator.input.creator.input, a)
        self.assertEqual(y.creator.input.creator.input.creator, sq_1)
        self.assertEqual(y.creator.input.creator.input.creator.input, x)

        # 手動でbackwardしていった場合
        y.grad = np.array(1.0)
        b.grad = y.creator.backward(y.grad)
        a.grad = b.creator.backward(b.grad)
        hand_grad = a.creator.backward(a.grad)

        # 自動でbackwardしていった場合
        y.backward()
        self.assertEqual(hand_grad, x.grad)

    def test_step09(self):
        x = Variable(np.array(0.5))
        a = square(x)
        b = exp(a)
        y = square(b)

        y.grad = np.array(1.0)
        y.backward()
        self.assertEqual(x.grad, 3.297442541400256)

    def test_step09_2(self):
        x = Variable(np.array(0.5))
        a = square(x)
        b = exp(a)
        y = square(b)

        y.backward()
        self.assertEqual(x.grad, 3.297442541400256)


if __name__ == '__main__':
    unittest.main()
