"""
falcon.py
Functional Linear Algebra with SEJITS
"""

import numpy as np
from falcon_array import FalconArray
from ctree.c.nodes import CFile
from ctree.cpp.nodes import CppInclude
from ctree.types import get_c_type_from_numpy_dtype
from ctypes import CFUNCTYPE
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.nodes import Project
from ctree.c.nodes import FunctionCall, FunctionDecl, SymbolRef

ENTRY_NAME = 'sejits_main'


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


class CFunction(ConcreteSpecializedFunction):
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

        print "Args: ", args

        # TODO: Return the c function INSTEAD!
        if self.c_func is None:
            python_output = self.py_func(*args)
            final_c_file = self.c_file_builder.build()

            # Get pointers for all the arguments
            argument_pointers = []
            for i, arg in enumerate(args):
                if isinstance(arg, FalconArray):
                    ptr = np.ctypeslib.ndpointer(args[i].dtype, args[i].ndim, args[i].shape)
                    argument_pointers.append(ptr)
                else:
                    scalar_data_type = get_c_type_from_numpy_dtype(np.dtype(type(arg)))()
                    argument_pointers.append(scalar_data_type)

            output_arr = np.zeros(python_output.shape).astype(python_output.dtype)
            argument_pointers.append(np.ctypeslib.ndpointer(output_arr.dtype,
                                                            output_arr.ndim, output_arr.shape))

            # TODO: need to deal with the pointer passing here; should not be argument_pointers[0]
            main = self.transf(final_c_file, argument_pointers[0])

            # TODO: make this into a helper function
            included_headers = []
            filtered_final_c_file = []
            for component in main.body:
                if isinstance(component, CppInclude):
                    already_found = False
                    for cpp in included_headers:
                        if cpp.target == component.target and \
                           cpp.angled_brackets == component.angled_brackets:
                            already_found = True
                            break

                    if not already_found:
                        included_headers.append(component)
                        filtered_final_c_file.append(component)
                else:
                    filtered_final_c_file.append(component)

            args = args + (output_arr, )
            main.body = filtered_final_c_file
            entry_type = CFUNCTYPE(None, *argument_pointers)
            self.c_func = self._compile(ENTRY_NAME, Project([main]), entry_type)
            self.c_func(*args)
            result = output_arr
        else:

            # TODO: the output array needs to be made of variable
            output_arr = np.zeros(args[0].shape).astype(args[0].dtype)
            args = args + (output_arr, )
            self.c_func(*args)
            result = output_arr
        return result

    def args_to_subconfig(self, args):
        raise NotImplementedError()

    def transf(self, c_file, pointer):

        print "Body: ", c_file.body

        body = []

        headers = [
            CppInclude("omp.h"),
            CppInclude("stdio.h")
        ]

        sejits_main = [FunctionDecl(None, ENTRY_NAME,
                                    params=[
                                        SymbolRef("input1", pointer()),
                                        SymbolRef("input2", pointer()),
                                        SymbolRef("output", pointer())
                                    ],
                                    defn=[
                                        # TODO: element_op, element_op2 need to be passed in to the
                                        # specializer
                                        FunctionCall('element_op', ['input1', 'input2', 'output']),
                                        FunctionCall('element_op2', ['output', 2, 'output'])
                                    ])
                       # FunctionCall() for statement in c_file.body if isinstance(statement,
                       # FunctionDecl)
                       ]

        body.extend(headers)
        body.extend(c_file.body)
        body.extend(sejits_main)

        final_c_file = CFile('generated_final', body, 'omp')
        return final_c_file

# TODO: We have to do more than just string together CFile objects, we have to actually compose


class CFileBuilder:
    """
    Helps build a CFile with a lot of functions
    """

    def __init__(self):
        self._build_complete = False
        self._c_files = []
        self._final_c_file = None

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
        new_body = []
        for file in self._c_files:
            new_body.extend(file.body)

        self._final_c_file = CFile("generated_final", new_body, "omp")
        return self._final_c_file


# This is an example function
def specialize(func):
    """
    Enables the specialization of a function.

    :param: func The function to decorate
    """
    new_func = CFunction(func)
    return new_func


# @specialize
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

    print "Number of C Files: ", len(double_input.c_file_builder._c_files)

    # print "Addition Result: ", test_inpt1 + test_inpt2
    # print "Subtraction Result: ", test_inpt1 - test_inpt2
    # print "Element-wise Multiplication Result: ", test_inpt1 * test_inpt2
    # print "Element-wise Division Result: ", test_inpt1 / test_inpt2
