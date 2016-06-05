"""
falcon.py
Functional Linear Algebra with SEJITS
"""

import numpy as np
from elem_wise_arr_op import specialize_element_wise

# Syntax

def dcRem(input):  # input has to be one of our objects
    avg = sum(input) / len(input)
    return input - avg


def specialize(func):
    """
    Enables the specialization of a function.

    :param: func The function to decorate
    """
    builder = CFileBuilder()  # the builder that builds things on the first call (JIT)
    # ... and continuing ...?
    return func  # <-- TODO actually do a c_func...


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
        assert isinstance(other, Scalar), "Operations on non-Scalar objects is not supported"
        return Scalar(self.value + other.value)     # TODO: Call a SEJITS function...

    def __sub__(self, other):
        assert isinstance(other, Scalar), "Operations on non-Scalar objects is not supported"
        return Scalar(self.value - other.value)     # TODO: Call a SEJITS function...

    def __mul__(self, other):
        assert isinstance(other, Scalar), "Operations on non-Scalar objects is not supported"
        return Scalar(self.value * other.value)     # TODO: Call a SEJITS function...

    def __div__(self, other):
        assert isinstance(other, Scalar), "Operations on non-Scalar objects is not supported"
        return Scalar(self.value / other.value)     # TODO: Call a SEJITS function...

    def __pow__(self, other, modulo):
        assert isinstance(other, Scalar), "Operations on non-Scalar objects is not supported"
        assert isinstance(modulo, Scalar), "Operations on non-Scalar objects is not supported"
        return Scalar(self.value ** other.value % modulo.value)


class FalconArray(np.ndarray):

    @staticmethod
    def array(*args, **kwargs):
        return np.array(*args, **kwargs).view(FalconArray)

    def __add__(self, other):
        return FalconArray.add(self, other)

    @staticmethod
    @specialize_element_wise
    def add(a, b):
        return a + b

    def __sub__(self, other):
        return FalconArray.subtract(self, other)

    @staticmethod
    @specialize_element_wise
    def subtract(a, b):
        return a - b

    def __mul__(self, other):
        if isinstance(other, FalconArray):
            assert self.shape == other.shape, \
                "Illegal element-wise multiplication with FalconArrays with shape {0} and {1}" \
                .format(self.shape, other.shape)

            return FalconArray.mul_elem_wise(self, other)
        elif isinstance(other, Scalar):
            return FalconArray(super.__mul__(self, other.value))

    @staticmethod
    @specialize_element_wise
    def mul_elem_wise(a, b):
        return a * b

    def __div__(self, other):
        if isinstance(other, FalconArray):
            assert self.shape == other.shape, \
                "Illegal element-wise division with FalconArrays with shape {0} and {1}" \
                .format(self.shape, other.shape)

            return FalconArray.div_elem_wise(self, other)
        elif isinstance(other, Scalar):
            return FalconArray(super.__div__(self, other.value))

    @staticmethod
    @specialize_element_wise
    def div_elem_wise(a, b):
        return a / b


#
# Testing Code
#

if __name__ == '__main__':
    TEST_INPT_LEN = 5
    test_inpt1 = FalconArray.array([2.0] * TEST_INPT_LEN)
    test_inpt2 = FalconArray.array([5.0] * TEST_INPT_LEN)

    print "Addition Result: ", test_inpt1 + test_inpt2
    print "Subtraction Result: ", test_inpt1 - test_inpt2
    print "Element-wise Multiplication Result: ", test_inpt1 * test_inpt2
    print "Element-wise Division Result: ", test_inpt1 / test_inpt2
