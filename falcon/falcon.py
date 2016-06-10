"""
falcon.py
Functional Linear Algebra with SEJITS
"""

import numpy as np
from elem_wise_arr_arr_op import specialize_arr_arr_element_wise
from elem_wise_arr_scalar_op import specialize_arr_scalar_element_wise


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

    #
    # Addition
    #

    def __add__(self, other):
        if isinstance(other, FalconArray):
            assert self.shape == other.shape, \
                "Illegal element-wise multiplication with FalconArrays of shape {0} and {1}" \
                .format(self.shape, other.shape)
            return FalconArray.add_arr_arr_elem_wise(self, other)
        else:
            return FalconArray.add_arr_scalar_elem_wise(self, other)

    @staticmethod
    @specialize_arr_arr_element_wise
    def add_arr_arr_elem_wise(a, b):
        """
        Performs element-wise addition on two input arrays.

        :param: a The array
        :param: b The array
        """
        return a + b

    @staticmethod
    @specialize_arr_scalar_element_wise
    def add_arr_scalar_elem_wise(a, b):
        """
        Performs element-wise addition on an input array.

        :param: a The array
        :param: b The scalar
        """
        return a + b

    #
    # Subtraction
    #

    def __sub__(self, other):
        if isinstance(other, FalconArray):
            assert self.shape == other.shape, \
                "Illegal element-wise multiplication with FalconArrays of shape {0} and {1}" \
                .format(self.shape, other.shape)
            return FalconArray.sub_arr_arr_elem_wise(self, other)
        else:
            return FalconArray.sub_arr_scalar_elem_wise(self, other)

    @staticmethod
    @specialize_arr_arr_element_wise
    def sub_arr_arr_elem_wise(a, b):
        """
        Performs element-wise subtraction on two input arrays.

        :param: a Input array
        :param: b Input array
        """
        return a - b

    @staticmethod
    @specialize_arr_scalar_element_wise
    def sub_arr_scalar_elem_wise(a, b):
        """
        Performs element-wise subtraction on an input array.

        :param: a The array
        :param: b The scalar
        """
        return a - b

    #
    # Multiplication
    #

    def __mul__(self, other):
        if isinstance(other, FalconArray):
            assert self.shape == other.shape, \
                "Illegal element-wise multiplication with FalconArrays of shape {0} and {1}" \
                .format(self.shape, other.shape)

            return FalconArray.mul_arr_arr_elem_wise(self, other)
        else:
            return FalconArray.mul_arr_scalar_elem_wise(self, other)

    @staticmethod
    @specialize_arr_arr_element_wise
    def mul_arr_arr_elem_wise(a, b):
        """
        Performs element-wise multiplication on two input arrays.

        :param: a Input array
        :param: b Input array
        """
        return a * b

    @staticmethod
    @specialize_arr_scalar_element_wise
    def mul_arr_scalar_elem_wise(a, b):
        """
        Performs element-wise multiplication on an input array.

        :param: a The array
        :param: b The scalar
        """
        return a * b

    #
    # Division
    #

    def __div__(self, other):
        if isinstance(other, FalconArray):
            assert self.shape == other.shape, \
                "Illegal element-wise division with FalconArrays of shape {0} and {1}" \
                .format(self.shape, other.shape)

            return FalconArray.div_arr_arr_elem_wise(self, other)
        else:
            return FalconArray.div_arr_scalar_elem_wise(self, other)

    @staticmethod
    @specialize_arr_arr_element_wise
    def div_arr_arr_elem_wise(a, b):
        """
        Performs element-wise division on two input arrays.

        :param: a Input array
        :param: b Input array
        """
        return a / b

    @staticmethod
    @specialize_arr_scalar_element_wise
    def div_arr_scalar_elem_wise(a, b):
        """
        Performs element-wise division on an input array.

        :param: a The array
        :param: b The scalar
        """
        return a / b

    #
    # Power (Not completed)
    #

    def __pow__(self, other):
        raise NotImplementedError

    #
    # Creation Methods
    #

    @staticmethod
    def empty(*args, **kwargs):
        return np.empty(*args, **kwargs).view(FalconArray)

    @staticmethod
    def zeros(*args, **kwargs):
        return np.zeros(*args, **kwargs).view(FalconArray)

    @staticmethod
    def zeros_like(*args, **kwargs):
        return np.zeros_like(*args, **kwargs).view(FalconArray)

    @staticmethod
    def rand(*args, **kwargs):
        return np.random.rand(*args, **kwargs).view(FalconArray)

    @staticmethod
    def standard_normal(*args, **kwargs):
        return np.random.standard_normal(*args, **kwargs).view(FalconArray)

    @staticmethod
    def empty_like(*args, **kwargs):
        return np.empty_like(*args, **kwargs).view(FalconArray)

    @staticmethod
    def ones(*args, **kwargs):
        return np.ones(*args, **kwargs).view(FalconArray)

    @staticmethod
    def array(*args, **kwargs):
        return np.array(*args, **kwargs).view(FalconArray)

    @staticmethod
    def ones_like(*args, **kwargs):
        return np.ones_like(*args, **kwargs).view(FalconArray)

    @staticmethod
    def eye(*args, **kwargs):
        return np.eye(*args, **kwargs).view(FalconArray)

    @staticmethod
    def fromstring(*args, **kwargs):
        return np.fromstring(*args, **kwargs).view(FalconArray)


#
# Testing Code
#

if __name__ == '__main__':
    TEST_INPT_LEN = 500000
    test_inpt1 = FalconArray.array([2.0] * TEST_INPT_LEN)
    test_inpt2 = FalconArray.array([5.0] * TEST_INPT_LEN)

    print "Test Output: ", dcRem(test_inpt1)


    # print "Addition Result: ", test_inpt1 + test_inpt2
    # print "Subtraction Result: ", test_inpt1 - test_inpt2
    # print "Element-wise Multiplication Result: ", test_inpt1 * test_inpt2
    # print "Element-wise Division Result: ", test_inpt1 / test_inpt2
