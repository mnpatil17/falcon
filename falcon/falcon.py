"""
falcon.py
Functional Linear Algebra with SEJITS
"""

import numpy as np
from elem_wise_arr_arr_op import specialize_arr_arr_element_wise
from elem_wise_arr_scalar_op import specialize_arr_scalar_element_wise

# TODO: We have to do more than just string together CFile objects, we have to actually compose
class CFileBuilder:
    """
    Helps build a CFile with a lot of functions
    """

    def __init__(self):
        self._build_complete = False
        self._c_files = []

    def add(self, new_c_files):          # Is this really supposed to be called "c_function?"
        """
        Adds a function to the C file.
        """
        self._c_files.extend(new_c_files)

    def build(self):
        """
        Finishes the building of the CFile object

        :return: A CFile object that contains the desired code
        """
        self._build_complete = True
        return self._c_files


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

            result = FalconArray.array(FalconArray.add_arr_arr_elem_wise(self, other))
            result._add_to_c_files_for_array = self._add_to_c_files_for_array
            self._add_to_c_files_for_array(
                FalconArray.add_arr_arr_elem_wise.latest_transform_result)

        else:
            result = FalconArray.array(FalconArray.add_arr_scalar_elem_wise(self, other))
            result._add_to_c_files_for_array = self._add_to_c_files_for_array
            self._add_to_c_files_for_array(
                FalconArray.add_arr_scalar_elem_wise.latest_transform_result)
        return result

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

            result = FalconArray.array(FalconArray.sub_arr_arr_elem_wise(self, other))
            result._add_to_c_files_for_array = self._add_to_c_files_for_array
            self._add_to_c_files_for_array(
                FalconArray.sub_arr_arr_elem_wise.latest_transform_result)
        else:
            result = FalconArray.array(FalconArray.sub_arr_scalar_elem_wise(self, other))
            result._add_to_c_files_for_array = self._add_to_c_files_for_array
            self._add_to_c_files_for_array(
                FalconArray.sub_arr_scalar_elem_wise.latest_transform_result)
        return result

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

            result = FalconArray.array(FalconArray.mul_arr_arr_elem_wise(self, other))
            result._add_to_c_files_for_array = self._add_to_c_files_for_array
            self._add_to_c_files_for_array(
                FalconArray.mul_arr_arr_elem_wise.latest_transform_result)
        else:
            result = FalconArray.array(FalconArray.mul_arr_scalar_elem_wise(self, other))
            result._add_to_c_files_for_array = self._add_to_c_files_for_array
            self._add_to_c_files_for_array(
                FalconArray.mul_arr_scalar_elem_wise.latest_transform_result)
        return result

    __rmul__ = __mul__
    __radd__ = __add__
    # TODO: __rsub__ and __rdiv__

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

            result = FalconArray.array(FalconArray.div_arr_arr_elem_wise(self, other))
            result._add_to_c_files_for_array = self._add_to_c_files_for_array
            self._add_to_c_files_for_array(
                FalconArray.div_arr_arr_elem_wise.latest_transform_result)
        else:
            result = FalconArray.array(FalconArray.div_arr_scalar_elem_wise(self, other))
            result._add_to_c_files_for_array = self._add_to_c_files_for_array
            self._add_to_c_files_for_array(
                FalconArray.div_arr_scalar_elem_wise.latest_transform_result)

        return result

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
    # Internal Methods
    #

    def _add_to_c_files_for_array(self, file_list):
        """
        This method should be re-assigned at some point before this FalconArray is used in
        operations.

        :param file_list: The list of files to add to the list of C files for the FalconArray
        """
        raise NotImplementedError()

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


class CFunction():
    """
    A primitive definition of a C Function that, when called, evaluates entirely in C

    FIXME: establish the guanrantee made above
    """

    def __init__(self, func):
        self.c_file_builder = CFileBuilder()
        self.py_func = func
        self.c_func = None

    def __call__(self, *args):
        def add_to_files(files):
            self.c_file_builder.add(files)

        for arg in args:
            if isinstance(arg, FalconArray):
                arg._add_to_c_files_for_array = add_to_files

        # TODO: Return the c function INSTEAD!
        return self.py_func(*args)


# This is an example function
def specialize(func):
    """
    Enables the specialization of a function.

    :param: func The function to decorate
    """
    new_func = CFunction(func)
    return new_func

@specialize
def dcRem(input):  # input has to be one of our objects
    avg = sum(input) / len(input)
    return input - avg


# This is another example function
@specialize
def double_input(input1, input2):
    diff = input1 - input2
    print type(diff)
    return 2 * diff


#
# Testing Code
#

if __name__ == '__main__':
    TEST_INPT_LEN = 500000
    test_inpt1 = FalconArray.array([2.0] * TEST_INPT_LEN)
    test_inpt2 = FalconArray.array([5.0] * TEST_INPT_LEN)

    print "Test Output: ", double_input(test_inpt1, test_inpt2)

    print len(double_input.c_file_builder._c_files)

    # print "Addition Result: ", test_inpt1 + test_inpt2
    # print "Subtraction Result: ", test_inpt1 - test_inpt2
    # print "Element-wise Multiplication Result: ", test_inpt1 * test_inpt2
    # print "Element-wise Division Result: ", test_inpt1 / test_inpt2
