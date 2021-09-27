# -*- coding: utf-8 -*-
"""
Utilitary functions for mathematical programming of complex vectors with NumPy
"""
import numpy as np
la = np.linalg


class complexarray(np.ndarray):
    """ Two features:
     - Allows to call Hermitian transpose of array x as x.H
       (as suggested in https://stackoverflow.com/questions/26932461/conjugate-transpose-operator-h-in-numpy)
     - quick self.encode method to output [real, imag] array
    """

    @property
    def H(self):
        return self.conj().T

    def encode(self):
        return np.concatenate((self.real, self.imag))


def decode_complex(realimag):
    realimag = np.asarray(realimag)
    NN = len(realimag)
    assert not NN % 2
    N = int(NN / 2)
    z = realimag[:N] + 1j * realimag[N:]
    return z.view(complexarray)


class MathError(Exception):
    pass
