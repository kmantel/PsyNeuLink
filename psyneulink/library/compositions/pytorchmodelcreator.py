from psyneulink.core.components.functions.transferfunctions import Linear, Logistic
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core import llvm as pnlvm
from llvmlite import ir
import numpy
import ctypes
import functools
import timeit
try:
    import torch
    from torch import nn
    torch_available = True
except ImportError:
    torch_available = False

import numpy as np

totaltime = 0
__all__ = ['PytorchModelCreator']
# Class that is called to create pytorch representations of autodiff compositions based on their processing graphs.
# Called to do so when the composition is run for the first time.

# Note on notation: the "nodes" that are constantly referred to are vertices of the composition's processing
# graph. For general compositions, the component a node represents can be a mechanism or a nested composition,
# but for autodiff compositions, nodes always represent mechanisms. "Nodes" can be thought of as
# (but are not literally) mechanisms.


class PytorchModelCreator(torch.nn.Module):

    # sets up parameters of model & the information required for forward computation
    def __init__(self, processing_graph, param_init_from_pnl, execution_sets, device, execution_id=None):

        if not torch_available:
            raise Exception('Pytorch python module (torch) is not installed. Please install it with '
                            '`pip install torch` or `pip3 install torch`')

        super(PytorchModelCreator, self).__init__()

        self.execution_sets = execution_sets  # saved for use in the forward method
        self.component_to_forward_info = {}  # dict mapping PNL nodes to their forward computation information
        self.projections_to_pytorch_weights = {}  # dict mapping PNL projections to Pytorch weights
        self.mechanisms_to_pytorch_biases = {}  # dict mapping PNL mechanisms to Pytorch biases
        self.params = nn.ParameterList()  # list that Pytorch optimizers will use to keep track of parameters
        self.device = device
        self.__bin_exec_func = None
        self._forward_llvm_func = None
        # Get first component in network
        inp_comp = self.execution_sets[0]
        # Get any mechanism
        inp_mech = min(inp_comp)

        # Get last component in network
        out_comp = self.execution_sets[len(self.execution_sets)-1]
        # Get any mechanism
        out_mech = min(out_comp)

        self.defaults = { # each of these should be of type array([[ ]]) for some reason
            # Assign input shape
            'variable':inp_mech.variable,
            # Assign output shape
            'value':out_mech.value
        }
        self._id_map = {}
        #print("INPMECH",inp_mech.variable,"OUTMECH",out_mech.value)
        for i in range(len(self.execution_sets)):
            # SKG: We have to add a counter to map components to an internal id (for bin execute purposes, since there is no concept of a 'dict' in llvm)
            comp_counter = 0
            self._id_map[i] = {}
            for component in self.execution_sets[i]:
                self._id_map[i][component] = comp_counter
                comp_counter += 1

                value = None  # the node's (its mechanism's) value
                biases = None  # the node's bias parameters
                function = self.function_creator(component, execution_id)  # the node's function
                afferents = {}  # dict for keeping track of afferent nodes and their connecting weights
                
                if param_init_from_pnl:
                    if component.parameters.value._get(execution_id) is None:
                        value = torch.tensor(component.parameters.value._get(None)[0])
                    else:
                        value = torch.tensor(component.parameters.value._get(execution_id)[0])
                else:
                    input_length = len(component.input_states[0].parameters.value._get(None))
                    value = torch.zeros(input_length, device=self.device).double()

                # if `node` is not an origin node (origin nodes don't have biases or afferent connections)
                if i != 0:
                    # if not copying parameters from psyneulink, set up pytorch biases for node
                    if not param_init_from_pnl:
                        input_length = len(component.input_states[0].parameters.value._get(None))
                        biases = nn.Parameter(torch.zeros(input_length, device=self.device).double())
                        self.params.append(biases)
                        self.mechanisms_to_pytorch_biases[component] = biases

                    # iterate over incoming projections and set up pytorch weights for them
                    for k in range(len(component.path_afferents)):

                        # get projection, sender node for projection
                        mapping_proj = component.path_afferents[k]
                        input_component = mapping_proj.sender.owner
                        input_node = processing_graph.comp_to_vertex[input_component]

                        # CW 12/3/18: Check this logic later
                        proj_matrix = mapping_proj.parameters.matrix._get(execution_id)
                        if proj_matrix is None:
                            proj_matrix = mapping_proj.parameters.matrix._get(None)

                        # set up pytorch weights that correspond to projection. If copying params from psyneulink,
                        # copy weight values from projection. Otherwise, use random values.
                        if param_init_from_pnl:
                            weights = nn.Parameter(torch.tensor(proj_matrix.copy(), device=self.device).double())
                        else:
                            weights = nn.Parameter(torch.rand(np.shape(proj_matrix), device=self.device).double())
                        afferents[input_node] = weights
                        self.params.append(weights)
                        self.projections_to_pytorch_weights[mapping_proj] = weights

                node_forward_info = [value, biases, function, afferents,component]
                # node_forward_info = [value, biases, function, afferents, value]

                self.component_to_forward_info[component] = node_forward_info


        # CW 12/3/18: this copies by reference so it only needs to be called during init, rather than
        # every time the weights are updated
        self.copy_weights_to_psyneulink(execution_id)
        self.__bin_exec_func = pnlvm.LLVMBinaryFunction.from_obj(self)

    # defines input type
    def _get_input_struct_type(self,ctx): # Test case: {[1 x [2 x double]]}
        struct_ty = ir.types.LiteralStructType([
            ctx.convert_python_struct_to_llvm_ir(self.defaults['variable'])
        ])
        #print("INP TYPE",struct_ty)
        return struct_ty
    
    # Converts tensor input to ctype
    def _get_input_struct(self,input,bin_func):
        inp_cty = bin_func.byref_arg_types[2]
        vals = None
        #TODO: allow multiple inputs at once
        for k,v in input.items():
            vals = v.numpy()
        vals = pnlvm.execution._tupleize([vals])
        #print("GOT INPUT STRUCT:",vals)
        return inp_cty(vals)

    def _get_output_struct_type(self,ctx): # Test case: {[1 x [1 x double]]}
        struct_ty = ir.types.LiteralStructType([
            ctx.convert_python_struct_to_llvm_ir(node.value) for node in self.execution_sets[len(self.execution_sets)-1]
        ])
        print("OUT TYPE",struct_ty)
        return struct_ty
    
    # Creates an output struct to be passed into forward computation
    def _get_output_struct(self,bin_func):
        outp_cty = bin_func.byref_arg_types[3]
        output = []
        for node in self.execution_sets[len(self.execution_sets) - 1]:
            output.append(node.value)
        output = pnlvm.execution._tupleize(output)
        #print("GET OUTPUT STRUCT",output,outp_cty)
        return outp_cty(output[0])
        
    def _get_param_struct_type(self,ctx):
        struct_ty = ir.types.LiteralStructType([
            ctx.convert_python_struct_to_llvm_ir(self.defaults['value'])
        ])
        #print("OUT TYPE",struct_ty)
        return struct_ty

    def _get_context_struct_type(self,ctx):
        struct_ty = ir.types.LiteralStructType([
            ctx.convert_python_struct_to_llvm_ir(self.defaults['value'])
        ])
        #print("OUT TYPE",struct_ty)
        return struct_ty

    # generates llvm function for self.forward (#TODO: allow forward func llvm to take weights as input, removing rebuild step)
    def _gen_llvm_function(self):
        llvm_func = None
        with pnlvm.LLVMBuilderContext.get_global() as ctx:
            args = [ctx.get_input_struct_type(self).as_pointer(),
                    ctx.get_input_struct_type(self).as_pointer(),
                    ctx.get_input_struct_type(self).as_pointer(),
                    ctx.get_output_struct_type(self).as_pointer()
                        ]
            #print("LLVM_FUNC ARGS:",args)
            builder = ctx.create_llvm_function(args,self) #NEED TO GET INPUT VECTOR SIZE (maybe just pass as ir.VectorType?)
            #print("BUILDER:",builder)
            llvm_func = builder.function
            #llvm_func.attributes.add('alwaysinline')
            
            params, context, arg_in, arg_out = llvm_func.args[:len(args)]
            self._gen_llvm_forward_function_body(ctx,builder,params,context,arg_in,arg_out)
            builder.ret_void()
            llvm_func = builder.function
            #print("LLVM_FUNC:\n",llvm_func,"\nDONE")
        self._forward_llvm_func = llvm_func
        return llvm_func

    # Injects matrix multiplication of m1 and m2 into builder (and returns allocated output)
    # m1 is x by y, m2 is y by z 
    def _gen_inject_matmul(self,ctx,builder,m1,m2,x,y,z):
        #print("m1:",m1,"m2:",m2)
        #create output mat
        output_mat = builder.alloca(ir.types.ArrayType(ir.types.ArrayType(ir.types.DoubleType(),z),x))
        # do computation
        for _x in range(0,x):
            #this is the current row we are making in the output
            for _z in range(0,z):
                curr_val = ir.types.DoubleType()(0)#
                for _y in range(0,y):
                    v1 = builder.load(builder.gep(m1,[ctx.int32_ty(0),ctx.int32_ty(_x),ctx.int32_ty(_y)]))
                    v2 = builder.load(builder.gep(m2,[ctx.int32_ty(0),ctx.int32_ty(_y),ctx.int32_ty(_z)]))
                    curr_val = builder.fadd(curr_val, builder.fmul(v1,v2))
                builder.store(curr_val,builder.gep(output_mat,[ctx.int32_ty(0),ctx.int32_ty(_x),ctx.int32_ty(_z)]))
                #builder.insert_value(builder.extract_value(builder.load(output_mat),_x),curr_val,_z)
        return output_mat

    # figure out type of arg_in
    def _gen_llvm_forward_function_body(self, ctx, builder, params, _, arg_in, arg_out):
        #print("ARGIN:",arg_in,"\nARGOUT:",arg_out)
        
        # Sometimes we arg_out to 2d array
        out_t = arg_out.type.pointee
        if isinstance(out_t, pnlvm.ir.ArrayType) and isinstance(out_t.element, pnlvm.ir.ArrayType):
            assert len(out_t) == 1
            # arg_out is a pointer to the beginning of the output array (TODO: add support for > 1 state)
            #arg_out = builder.gep(arg_out, [ctx.int32_ty(0), ctx.int32_ty(0), ctx.int32_ty(0)])
        #allocate stack values for frozen values (organized in dict for ease of use)
        frozen_values = {}
        for i in range(len(self.execution_sets)):
            current_exec_set = self.execution_sets[i]
            
            for component in current_exec_set:
                frozen_values[component] = builder.alloca(ctx.convert_python_struct_to_llvm_ir(component.variable))
                print("alloca ",component.variable,"frozen val:",frozen_values[component],"\n component",component)
            for component in current_exec_set:
                biases = self.component_to_forward_info[component][1]
                value = frozen_values[component]
                afferents = self.component_to_forward_info[component][3] #TODO: Pass in afferents as input to func
                dim_x,dim_y = component.variable.shape
                if i == 0:
                    for _x in range(0,dim_x):
                        for _y in range(0,dim_y):
                            cmp_arg = builder.gep(arg_in,[ctx.int32_ty(0),ctx.int32_ty(0),ctx.int32_ty(_x),ctx.int32_ty(_y)]) # get input
                            res = self.bin_function_creator(ctx,builder,component,builder.load(cmp_arg),execution_id=None)
                            builder.store(res,builder.gep(value,[ctx.int32_ty(0),ctx.int32_ty(_x),ctx.int32_ty(_y)]))
                else:
                    # is_set keeps track of if we already have valid (i.e. non-garbage) values inside the alloc'd value
                    is_set = False
                    for input_node, weights in afferents.items():
                        if input_node.component in frozen_values:
                            input_value = frozen_values[input_node.component]
                        else:
                            # Does this actually extract the value i want?
                            input_value = self.component_to_forward_info[input_node.component][0].numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                        matmul = ctx.get_llvm_function("__pnl_builtin_vxm")
                        #TODO: Make sure weights are dynamically updated every run of forward (so backpropagation works as intended)
                        # To do ^, we have to make sure that the weights reference stays the same (i.e. the np is mutable, instead of reinstantiated every time it is updated)
                        #Also TODO: Figure out what exactly detach does (how does autograd actually work internally)
                        weights_np = weights.detach().numpy()
                        x,y = weights_np.shape
                        # convert weights np array to ct
                        weights_ct = weights_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                        weights_pt = ctypes.cast(weights_ct, ctypes.c_void_p).value
                        #We cast the ctype weights array to llvmlite pointer (TODO: make sure this reference doesn't disappear due to GC!)
                        weights_llvmlite = ir.types.IntType(64)(weights_pt)
                        weights_llvmlite = builder.inttoptr(weights_llvmlite,ir.types.ArrayType(ir.types.ArrayType(ir.types.DoubleType(),y),x).as_pointer())
                        weighted_inp = self._gen_inject_matmul(ctx,builder,input_value,weights_llvmlite,1,x,y)
                        #print(weighted_inp)
                        if (i == 1):
                            #self._output_forward_computation(ctx,builder,arg_out,0,0,0,(builder.load(
                            #    builder.gep(weighted_inp,[ctx.int32_ty(0),ctx.int32_ty(0),ctx.int32_ty(0)])
                            #    )))
                            pass
                        if is_set == False:
                            # copy weighted_inp to value
                            for _x in range(0,1):
                                for _y in range(0,y):
                                    loc = builder.gep(value,[ctx.int32_ty(0),ctx.int32_ty(_x),ctx.int32_ty(_y)])
                                    val_ptr = builder.gep(weighted_inp,[ctx.int32_ty(0),ctx.int32_ty(_x),ctx.int32_ty(_y)])
                                    builder.store(builder.load(val_ptr),loc)
                            is_set = True
                        else:
                            # add to value
                            for _x in range(0,1):
                                for _y in range(0,y):
                                    loc = builder.gep(value,[ctx.int32_ty(0),ctx.int32_ty(_x),ctx.int32_ty(_y)])
                                    val_ptr = builder.gep(weighted_inp,[ctx.int32_ty(0),ctx.int32_ty(_x),ctx.int32_ty(_y)])
                                    builder.store(builder.fadd(builder.load(loc),builder.load(val_ptr)),loc)
                        #value += torch.matmul(input_value, weights)
                    for _x in range(0,dim_x):
                        for _y in range(0,dim_y):
                            cmp_arg = builder.gep(value,[ctx.int32_ty(0),ctx.int32_ty(_x),ctx.int32_ty(_y)]) # get input vector
                            res = self.bin_function_creator(ctx,builder,component,builder.load(cmp_arg),execution_id=None)
                            builder.store(res,builder.gep(value,[ctx.int32_ty(0),ctx.int32_ty(_x),ctx.int32_ty(_y)]))
                    #TODO: Add bias to value
                    #if biases is not None:
                    #   value = value + biases

                # save value in output list if we're at a node in the last execution set
                if i == len(self.execution_sets) - 1:
                    # We first grab which index we should insert into:
                    # Here, arr should be an array of arrays; each index correlates to self._id_map[i][component]
                    output_index = self._id_map[i][component]

                    # get ptr to first thing in struct (should be arr)
                    outer_arr_ptr = builder.gep(arg_out,[ctx.int32_ty(0), ctx.int32_ty(0)]) 
                   # print("OUTER ARR",outer_arr)
                    for _x in range(0,1):
                        for _y in range(0,1):
                            val_ptr = builder.gep(value,[ctx.int32_ty(0), ctx.int32_ty(_x), ctx.int32_ty(_y)])
                            self._output_forward_computation(ctx,builder,arg_out,0,_x,_y,builder.load(val_ptr))
                            

    # inserts a value into the forward computation output array struct
    def _output_forward_computation(self,ctx,builder,arg_out,index,x,y,value):
        loc = builder.gep(arg_out,[ctx.int32_ty(0), ctx.int32_ty(index),ctx.int32_ty(x), ctx.int32_ty(y)]) 
        builder.store(value,loc)
    
    @property
    def _bin_exec_func(self):        
        if self.__bin_exec_func is None:
            self.__bin_exec_func = pnlvm.LLVMBinaryFunction.from_obj(self)
        return self.__bin_exec_func
    
    # Creates a set that maps each component to its output (from llvm execution)
    # Uses self.id_map
    def _remap_output_struct(self,outputs):
        output_dict = {}
        for component in self.execution_sets[len(self.execution_sets)-1]:
            id = self._id_map[len(self.execution_sets)-1][component]
            output_dict[component] = torch.from_numpy(numpy.asarray(outputs[id]))
        return output_dict
    # performs forward computation for the model
    def forward(self, inputs, execution_id=None, do_logging=True,bin_execute=True):

        if bin_execute is True:
            print("INPUT:",inputs)
            for component in self.execution_sets[1]:
                afferents = self.component_to_forward_info[component][3]
                for input_node, weights in afferents.items():
                    print("TESTWEIGHT",weights.detach().numpy())
            global totaltime
            start_time = timeit.default_timer()
            bin_fun = self._bin_exec_func
            
            outputs = self._get_output_struct(bin_fun)
            bin_fun.wrap_call(self._get_input_struct(inputs,bin_fun),
                                self._get_input_struct(inputs,bin_fun),
                                self._get_input_struct(inputs,bin_fun),
                                outputs)
            totaltime += timeit.default_timer() - start_time
            outputs = pnlvm.execution._convert_ctype_to_python(outputs)
            print(totaltime,outputs)
            #print(self._remap_output_struct(outputs))
            #return self._remap_output_struct(outputs)
        outputs = {}  # dict for storing values of terminal (output) nodes

        for i in range(len(self.execution_sets)):
            current_exec_set = self.execution_sets[i]
            frozen_values = {}
            for component in current_exec_set:
                frozen_values[component] = self.component_to_forward_info[component][0]
            for component in current_exec_set:

                # get forward computation info for current component
                biases = self.component_to_forward_info[component][1]
                function = self.component_to_forward_info[component][2]
                afferents = self.component_to_forward_info[component][3]

                # forward computation if we have origin node
                if i == 0:
                    value = function(inputs[component])
                    #print("INIT",value,inputs)

                # forward computation if we do not have origin node
                else:
                    value = torch.zeros(len(component.input_states[0].defaults.value), device=self.device).double()
                    for input_node, weights in afferents.items():
                        if input_node.component in current_exec_set:
                            input_value = frozen_values[input_node.component]
                        else:
                            input_value = self.component_to_forward_info[input_node.component][0]
                        print("i:",i,"VAL:",value,"INP:",input_value.detach().numpy(),"MULT:",torch.matmul(input_value, weights),"BIASES:",biases)
                        value += torch.matmul(input_value, weights)
                    if biases is not None:
                        value = value + biases
                    value = function(value)

                # store the current value of the node
                self.component_to_forward_info[component][0] = value
                if do_logging:
                    detached_value = value.detach().numpy()
                    component.parameters.value._log_value(detached_value, execution_id, ContextFlags.COMMAND_LINE)

                # save value in output list if we're at a node in the last execution set
                if i == len(self.execution_sets) - 1:
                    outputs[component] = value

        self.copy_outputs_to_psyneulink(outputs, execution_id)
        if do_logging:
            self.log_weights(execution_id)
        print("\tACTUAL:",outputs,"\n")
        return outputs

    def detach_all(self):
        for component, info in self.component_to_forward_info.items():
            info[0].detach_()
            if info[1] is not None:
                info[1].detach_()

    # def reset_all(self):
    #     for component, info in self.component_to_forward_info.items():
    #         info[0] = info[4]

    def copy_weights_to_psyneulink(self, execution_id=None):
        for projection, weights in self.projections_to_pytorch_weights.items():
            projection.parameters.matrix._set(weights.detach().numpy(), execution_id)

    def copy_outputs_to_psyneulink(self, outputs, execution_id=None):
        for component, value in outputs.items():
            detached_value = value.detach().numpy()
            component.parameters.value._set(detached_value, execution_id, override=True, skip_history=True, skip_log=True)
            component.output_state.parameters.value._set(detached_value, execution_id, override=True, skip_history=True, skip_log=True)

    def log_weights(self, execution_id=None):
        for projection, weights in self.projections_to_pytorch_weights.items():
            projection.parameters.matrix._log_value(weights.detach().numpy(), execution_id, ContextFlags.COMMAND_LINE)
    # helper method that functions the same as function_creator, but instead injects the computation to the builder

    def bin_function_creator(self,ctx,builder,node,x,execution_id=None):
        ir_dbl = ir.types.DoubleType()
        
        def get_fct_param_value(param_name):
            val = node.function.get_current_function_param(param_name, execution_id)
            if val is None:
                val = node.function.get_current_function_param(param_name, None)
            return ir_dbl(val[0])

        if isinstance(node.function, Linear):
            slope = get_fct_param_value('slope')
            intercept = get_fct_param_value('intercept')
            
            ret = builder.fadd(builder.fmul(x,slope),intercept)
            return ret
            #builder.ret(ret)

        elif isinstance(node.function, Logistic):
            neg_one = ir_dbl(-1)
            gain = builder.fmul(neg_one,get_fct_param_value('gain'))
            bias = get_fct_param_value('bias')
            offset = get_fct_param_value('offset')
            one = ir_dbl(1)
            exp = ctx.get_llvm_function("__pnl_builtin_exp")
            arg = builder.fadd(
                builder.fmul(
                    gain, builder.fsub(x,bias)
                ),offset)
            
            ret = builder.fdiv(one,builder.fadd(one,builder.call(exp,[arg])))
            return ret
            #builder.ret(ret)
            #return lambda x: 1 / (1 + torch.exp(-gain * (x - bias) + offset))

        else:  # if we have relu function (the only other kind of function allowed by the autodiff composition)
            gain = get_fct_param_value('gain')
            bias = get_fct_param_value('bias')
            leak = get_fct_param_value('leak')
            zero = ir_dbl(0)
            val = builder.fsub(x,bias)
            pred = builder.fcmp_ordered(">",val,zero)

            with builder.if_else(pred) as (then, otherwise):
                with then:
                    max = val
                    min = zero
                with otherwise:
                    max = zero
                    min = val
            
            ret = builder.fadd(
                builder.fmul(max,gain),
                builder.fmul(min,leak))
            return ret
            #builder.ret(ret)
        #return builder.function

    # helper method that identifies the type of function used by a node, gets the function
    # parameters and uses them to create a function object representing the function, then returns it
    def function_creator(self, node, execution_id=None):
        def get_fct_param_value(param_name):
            val = node.function.get_current_function_param(param_name, execution_id)
            if val is None:
                val = node.function.get_current_function_param(param_name, None)
            return float(val)

        if isinstance(node.function, Linear):
            slope = get_fct_param_value('slope')
            intercept = get_fct_param_value('intercept')
            return lambda x: x * slope + intercept

        elif isinstance(node.function, Logistic):
            gain = get_fct_param_value('gain')
            bias = get_fct_param_value('bias')
            offset = get_fct_param_value('offset')
            return lambda x: 1 / (1 + torch.exp(-gain * (x - bias) + offset))

        else:  # if we have relu function (the only other kind of function allowed by the autodiff composition)
            gain = get_fct_param_value('gain')
            bias = get_fct_param_value('bias')
            leak = get_fct_param_value('leak')
            return lambda x: (torch.max(input=(x - bias), other=torch.tensor([0], device=self.device).double()) * gain +
                              torch.min(input=(x - bias), other=torch.tensor([0], device=self.device).double()) * leak)

    # returns dict mapping psyneulink projections to corresponding pytorch weights. Pytorch weights are copied
    # over from tensors inside Pytorch's Parameter data type to numpy arrays (and thus copied to a different
    # memory location). This keeps the weights - and Pytorch in general - away from the user
    def get_weights_for_projections(self):
        weights_in_numpy = {}
        for projection, weights in self.projections_to_pytorch_weights.items():
            weights_in_numpy[projection] = weights.detach().numpy().copy()
        return weights_in_numpy

    # returns dict mapping psyneulink mechanisms to corresponding pytorch biases, the same way as the above function.
    # If composition is initialized with "param_init_from_PNL" set to true, then no biases are created in Pytorch,
    # and when called, this function returns an empty list.
    def get_biases_for_mechanisms(self):
        biases_in_numpy = {}
        for mechanism, biases in self.mechanisms_to_pytorch_biases.items():
            biases_in_numpy[mechanism] = biases.detach().numpy().copy()
        return biases_in_numpy
