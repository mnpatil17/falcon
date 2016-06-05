"""
falcon.py
Functional Linear Algebra with SEJITS
"""

import numpy as np

# Syntax

@specialize
def dcRem(input): # input has to be one of our objects
    avg = sum(input) / len(input)
    return input - avg


def specialize(func):
    """
    Enables the specialization of a function.

    :param: func The function to decorate
    """
    builder = CFileBuilder() # the builder that builds things on the first call (JIT)
    # ... and continuing ...?
    return func # <-- TODO actually do a c_func...


# TODO: We have to do more than just string together CFile objects, we have to actually compose
class CFileBuilder:
    """
    Helps build a CFile with a lot of functions
    """

    def __init__(self):
        self._build_complete = False
        self._c_file = CFile()          # TODO: actually make an empty CFile object here...

    def add(self, c_function):          # Is this really supposed to be called "c_function?"
        """
        Adds a function to the C file.
        """
        raise NotImplementedError


    def build(self):
        """
        Finishes the building of the CFile object

        :return: A CFile object that contains the desired code
        """
        self._build_complete = True
        return self._c_file


class Scalar:
    """
    An implementation of SEJITS operations on scalar values.
    """

    def __init__(self, value):
        self.value = value

    def __add__(self, other):
        assert other isinstance Scalar, "Operations on non-Scalar objects is not supported"
        return Scalar(self.value + other.value)     # TODO: Call a SEJITS function...

    def __sub__(self, other):
        assert other isinstance Scalar, "Operations on non-Scalar objects is not supported"
        return Scalar(self.value - other.value)     # TODO: Call a SEJITS function...

    def __mul__(self, other):
        assert other isinstance Scalar, "Operations on non-Scalar objects is not supported"
        return Scalar(self.value * other.value)     # TODO: Call a SEJITS function...

    def __div__(self, other):
        assert other isinstance Scalar, "Operations on non-Scalar objects is not supported"
        return Scalar(self.value / other.value)     # TODO: Call a SEJITS function...

    def __pow__(self, other, modulo):
        assert other isinstance Scalar, "Operations on non-Scalar objects is not supported"
        assert modulo isinstance Scalar, "Operations on non-Scalar objects is not supported"
        return Scalar(self.value ** other.value % modulo.value)


class FalconArray(np.array):

    # we need to be able to get the C functions for all these things, somehow
    # def sum(self):
    #     return c_sum_func(self)  # TODO: needs to be replaced by a call to tbe backend

    def __add__(self, other):
        return self + other      # TODO: needs to be replaced by a call to the backend

    def __sub__(self, other):
        return self - other      # TODO: needs to be replaced by a call to the backend

    def __mul__(self, other):
        if other isinstance FalconArray:
            assert self.shape == other.shape,
                "Illegal element-wise multiplication with FalconArrays with shape {0} and {1}"
                .format(self.shape, other.shape)

            return FalconArray(super.__mul__(self, other))
        elif other isinstance Scalar:
            return FalconArray(super.__mul__(self, other.value))

    def __div__(self, other):
        if other isinstance FalconArray
            assert self.shape == other.shape,
                "Illegal element-wise division with FalconArrays with shape {0} and {1}"
                .format(self.shape, other.shape)

            return FalconArray(super.__div__(self, other))
        elif other isinstance Scalar:
            return FalconArray(super.__div__(self, other.value))

