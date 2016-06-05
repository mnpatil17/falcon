# A specializer to perform segmented reduction that parallelizes using the OpenMP Framework.

# Importations
from __future__ import print_function, division
import numpy as np
import ctypes as ct
import time

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


FUNC_NAME = 'element_op'


#
# Specializer Code
#


class ConcreteArrayOp(ConcreteSpecializedFunction):

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


class LazyArrayOp(LazySpecializedFunction):

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

        reducer = CFile("generated", [
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
        ], 'omp')  # <--- TODO: is this necessarily supposed to be OMP

        return [reducer]

    def finalize(self, transform_result, program_config):
        tree = transform_result[0]

        # Get the argument type data
        input_data = program_config[0]
        pointer = np.ctypeslib.ndpointer(input_data.dtype, input_data.ndim, input_data.shape)
        entry_type = CFUNCTYPE(None, pointer, pointer, pointer)

        # Instantiation of the concrete function
        fn = ConcreteArrayOp()

        return fn.finalize(Project([tree]), FUNC_NAME, entry_type)

if __name__ == '__main__':

    def add(x, y):
        return x + y

    def mul(x, y):
        return x * y

    def div(x, y):
        return x / y

    TEST_INPT_LEN = 5
    test_inpt1 = np.array([5.0] * TEST_INPT_LEN)
    test_inpt2 = np.array([2.0] * TEST_INPT_LEN)

    element_wise_array_op = LazyArrayOp.from_function(div, "ArrayOp")
    result = element_wise_array_op(test_inpt1, test_inpt2)

    print ("Result: ", result)
