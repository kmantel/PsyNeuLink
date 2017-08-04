#!/usr/bin/python3

import ctypes
import PsyNeuLink.llvm as pnlvm
import numpy as np
import copy
from llvmlite import ir

ITERATIONS=100
DIM_X=1000

matrix = np.random.rand(DIM_X, DIM_X)
vector = np.random.rand(DIM_X)
llvm_res = np.random.rand(DIM_X)

ct_vec = vector.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
ct_mat = matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
x, y = matrix.shape

# The original builtin mxv function
binf = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_vxm')
orig_res = copy.deepcopy(llvm_res)
ct_res = orig_res.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

binf.c_func(ct_vec, ct_mat, x, y, ct_res)

with pnlvm.llvm_get_current_ctx() as ctx:
    double_ptr_ty = ctx.float_ty.as_pointer()
    func_ty = ir.FunctionType(ir.VoidType(), (double_ptr_ty, double_ptr_ty, double_ptr_ty))

    # get builtin IR
    builtin = ctx.get_llvm_function('__pnl_builtin_vxm')

    # Create square vector matrix multiply
    function = ir.Function(ctx.module, func_ty, name="vxsqm")
    _x = ctx.int32_ty(x)
    _v, _m, _o = function.args
    block = function.append_basic_block(name="entry")
    builder = ir.IRBuilder(block)
    builder.call(builtin, [_v, _m, _x, _x, _o])
    builder.ret_void()

# Rebuild and try again
#pnlvm.llvm_build()

binf2 = pnlvm.LLVMBinaryFunction.get('vxsqm')
new_res = copy.deepcopy(llvm_res)
ct_res = new_res.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

binf2(ct_vec, ct_mat, ct_res)

if np.array_equal(orig_res, new_res):
    print("TEST PASSED")
else:
    print("TEST FAILED")

