import numpy as np
from elem_wise_arr_arr_op import specialize_arr_arr_element_wise
from elem_wise_arr_scalar_op import specialize_arr_scalar_element_wise


class FalconArray(np.ndarray):
    """
    An array that is SEJITS specialized.
    """

    #
    # Addition
    #

    def __add__(self, other):
        if isinstance(other, FalconArray):
            assert self.shape == other.shape, \
                "Illegal element-wise multiplication with FalconArrays of shape {0} and {1}" \
                .format(self.shape, other.shape)

            result = self.get_result_binary(other, FalconArray.add_arr_arr_elem_wise)
        else:
            result = self.get_result_binary(other, FalconArray.add_arr_scalar_elem_wise)
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

            result = self.get_result_binary(other, FalconArray.sub_arr_arr_elem_wise)
        else:
            result = self.get_result_binary(other, FalconArray.sub_arr_scalar_elem_wise)
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

            result = self.get_result_binary(other, FalconArray.mul_arr_arr_elem_wise)
        else:
            result = self.get_result_binary(other, FalconArray.mul_arr_scalar_elem_wise)
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

            result = self.get_result_binary(other, FalconArray.div_arr_arr_elem_wise)
        else:
            result = self.get_result_binary(other, FalconArray.div_arr_scalar_elem_wise)

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

    def get_result_binary(self, other, func):
        result = FalconArray.array(func(self, other))
        result._add_to_c_files_for_array = self._add_to_c_files_for_array
        self._add_to_c_files_for_array(func.latest_transform_result)
        return result

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
