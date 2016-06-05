"""
A specializer to perform element-wise array operations with the SEJITS Framework.
"""

# Importations
from __future__ import print_function, division
import numpy as np

from ctree.nodes import Project
from ctree.c.nodes import *
from ctree.c.macros import *
from ctree.cpp.nodes import *
from ctree.omp.nodes import *
from ctree.omp.macros import *
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctypes import CFUNCTYPE
from ctree.templates.nodes import StringTemplate

from ctree.transformations import PyBasicConversions
from ctree.types import get_c_type_from_numpy_dtype

from collections import namedtuple
import inspect


#
# Specializer Decorator
#


def specialize_element_wise(func):
    """
    Specializes a two argument function.

    :func: a two argument function that represents the element-wise array operation
    :return: a function that takes in 2 arrays and returns an output of the same length
    """
    func_name_capital = func.__name__[0].upper() + func.__name__[1:]
    func_hash = str(hash(func))
    num_func_args = len(inspect.getargspec(func).args)
    assert num_func_args == 2, \
        "Element-wise operations must take exactly two arguments; {0} given".format(num_func_args)

    return LazyElemWiseArrayOp.from_function(func, "ArrayOp" + func_name_capital + "_" + func_hash)


#
# Specializer Code
#


FUNC_NAME = 'element_op'


class ConcreteElemWiseArrayOp(ConcreteSpecializedFunction):
    """
    The actual python callable for DC Removal Specalizer.
    """

    def finalize(self, tree, entry_name, entry_type):
        self._c_function = self._compile(entry_name, tree, entry_type)
        return self

    def __call__(self, input1, input2):

        assert input1.shape == input2.shape, \
            "Both input arrays must have the same shape for element-wise operations"

        output_arr = np.zeros(input1.shape).astype(input1.dtype)
        self._c_function(input1, input2, output_arr)
        return output_arr


class LazyElemWiseArrayOp(LazySpecializedFunction):
    """
    The lazy version of the DC Removal Specializer that handles code generation just in time for
    execution.
    """
    subconfig_type = namedtuple('subconfig', ['dtype', 'ndim', 'shape', 'size', 'flags'])

    def args_to_subconfig(self, args):
        input1 = args[0]
        return self.subconfig_type(input1.dtype, input1.ndim, input1.shape, input1.size, [])

    def transform(self, py_ast, program_config):

        # Get the initial data
        input_data = program_config[0]
        length = np.prod(input_data.size)
        pointer = np.ctypeslib.ndpointer(input_data.dtype, input_data.ndim, input_data.shape)
        data_type = get_c_type_from_numpy_dtype(input_data.dtype)()

        apply_one = PyBasicConversions().visit(py_ast.body[0])
        apply_one.name = 'apply'
        apply_one.params[0].type = data_type
        apply_one.params[1].type = data_type
        apply_one.return_type = data_type

        array_add_template = StringTemplate(r"""
            #pragma omp parallel for
            for (int i = 0; i < $length; i++) {
                output[i] = apply(input1[i], input2[i]);
            }
        """, {
            'length': Constant(length)
        })

        array_op = CFile("generated", [
            CppInclude("omp.h"),
            CppInclude("stdio.h"),
            apply_one,
            FunctionDecl(None, FUNC_NAME,
                         params=[
                             SymbolRef("input1", pointer()),
                             SymbolRef("input2", pointer()),
                             SymbolRef("output", pointer())
                         ],
                         defn=[
                             array_add_template
                         ])
        ], 'omp')

        return [array_op]

    def finalize(self, transform_result, program_config):
        tree = transform_result[0]

        # Get the argument type data
        input_data = program_config[0]
        pointer = np.ctypeslib.ndpointer(input_data.dtype, input_data.ndim, input_data.shape)
        entry_type = CFUNCTYPE(None, pointer, pointer, pointer)

        # Instantiation of the concrete function
        fn = ConcreteElemWiseArrayOp()

        return fn.finalize(Project([tree]), FUNC_NAME, entry_type)


#
# Testing Code
#

if __name__ == '__main__':

    @specialize_element_wise
    def add(x, y):
        return x + y

    @specialize_element_wise
    def mul(x, y):
        return x * y

    @specialize_element_wise
    def div(x, y):
        return x / y

    TEST_INPT_LEN = 5
    test_inpt1 = np.array([5.0] * TEST_INPT_LEN)
    test_inpt2 = np.array([2.0] * TEST_INPT_LEN)

    add_result = add(test_inpt1, test_inpt2)
    mul_result = mul(test_inpt1, test_inpt2)
    div_result = div(test_inpt1, test_inpt2)

    print ("Added Correctly: ", all(add_result == test_inpt1 + test_inpt2))
    print ("Muliplied Correctly: ", all(mul_result == test_inpt1 * test_inpt2))
    print ("Divided Correctly: ", all(div_result == test_inpt1 / test_inpt2))
