import unittest

import numpy as np

from fw.stage02.exp import exp
from fw.stage02.square import square
from fw.stage02.variable import Variable


class Stage02TestCase(unittest.TestCase):
    def test_step12(self):
        xs = [Variable(s) for s in np.arange(0, 1, 0.5)]
        a = square(*xs)
        b = exp(*a)
        ys = square(*b)
        self.assertEqual([y.data for y in ys], [1.0, 1.648721270700128])


if __name__ == '__main__':
    unittest.main()
