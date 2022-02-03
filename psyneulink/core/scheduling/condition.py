# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* Condition **************************************************************

import collections
import copy
import functools
import inspect

import dill
import graph_scheduler

from psyneulink.core.globals.context import handle_external_context
from psyneulink.core.globals.json import JSONDumpable
from psyneulink.core.globals.keywords import MODEL_SPEC_ID_TYPE, comparison_operators
from psyneulink.core.globals.parameters import parse_context

__all__ = copy.copy(graph_scheduler.condition.__all__)
__all__.extend(['Threshold'])


def _create_as_pnl_condition(condition):
    import psyneulink as pnl

    try:
        pnl_class = getattr(pnl.core.scheduling.condition, type(condition).__name__)
    except (AttributeError, TypeError):
        return condition

    # already a pnl Condition
    if isinstance(condition, Condition):
        return condition

    if not issubclass(pnl_class, graph_scheduler.Condition):
        return None

    new_args = [_create_as_pnl_condition(a) or a for a in condition.args]
    new_kwargs = {k: _create_as_pnl_condition(v) or v for k, v in condition.kwargs.items()}
    sig = inspect.signature(pnl_class)

    if 'func' in sig.parameters or 'function' in sig.parameters:
        # Condition takes a function as an argument
        res = pnl_class(condition.func, *new_args, **new_kwargs)
    else:
        res = pnl_class(*new_args, **new_kwargs)

    res.owner = condition.owner
    return res


class Condition(graph_scheduler.Condition, JSONDumpable):
    @handle_external_context()
    def is_satisfied(self, *args, context=None, execution_id=None, **kwargs):
        if execution_id is None:
            try:
                execution_id = parse_context(context).execution_id
            except AttributeError:
                pass

        return super().is_satisfied(
            *args,
            context=context,
            execution_id=execution_id,
            **kwargs
        )

    @property
    def _dict_summary(self):
        from psyneulink.core.components.component import Component

        if type(self) is graph_scheduler.Condition:
            try:
                func_val = inspect.getsource(self.func)
            except OSError:
                func_val = dill.dumps(self.func)
        else:
            func_val = None

        args_list = []
        for a in self.args:
            if isinstance(a, Component):
                a = a.name
            elif isinstance(a, graph_scheduler.Condition):
                a = a._dict_summary
            args_list.append(a)

        return {
            MODEL_SPEC_ID_TYPE: self.__class__.__name__,
            'function': func_val,
            'args': args_list,
            'kwargs': self.kwargs,
        }


# below produces psyneulink versions of each Condition class so that
# they are compatible with the extra changes made in Condition above
# (the scheduler does not handle Context objects or mdf/json export)
cond_dependencies = {}
pnl_conditions_module = locals()  # inserting into locals defines the classes

for cond_name in graph_scheduler.condition.__all__:
    sched_module_cond_obj = getattr(graph_scheduler.condition, cond_name)
    cond_dependencies[cond_name] = set(sched_module_cond_obj.__mro__[1:])

# iterate in order such that superclass types are before subclass types
for cond_name in sorted(
    graph_scheduler.condition.__all__,
    key=functools.cmp_to_key(lambda a, b: -1 if b in cond_dependencies[a] else 1)
):
    # don't substitute Condition because it is explicitly defined above
    if cond_name == 'Condition':
        continue

    sched_module_cond_obj = getattr(graph_scheduler.condition, cond_name)
    if (
        inspect.isclass(sched_module_cond_obj)
        and issubclass(sched_module_cond_obj, graph_scheduler.Condition)
    ):
        new_mro = []
        for cls_ in sched_module_cond_obj.__mro__:
            if cls_ is not graph_scheduler.Condition:
                try:
                    new_mro.append(pnl_conditions_module[cls_.__name__])

                except KeyError:
                    new_mro.append(cls_)
            else:
                new_mro.extend(Condition.__mro__[:-1])
        pnl_conditions_module[cond_name] = type(cond_name, tuple(new_mro), {})
    elif isinstance(sched_module_cond_obj, type):
        pnl_conditions_module[cond_name] = sched_module_cond_obj

    pnl_conditions_module[cond_name].__doc__ = sched_module_cond_obj.__doc__


_doc_subs = {
    None: [
        (
            r'def converge\(node, thresh\):',
            'def converge(node, thresh, context):'
        ),
        (
            r'node\.delta',
            'node.parameters.value.get_delta(context)'
        ),
        (
            r'the ``delta`` attribute of\na node \(which reports the change in its ``value``\)',
            "the change in a node's `value`"
        )
    ]
}


class Threshold(graph_scheduler.condition._DependencyValidation, Condition):
    def __init__(
        self, dependency, parameter, threshold, comparator, indices=None
    ):
        if comparator not in comparison_operators:
            raise graph_scheduler.ConditionError(f'Operator must be one of {list(comparison_operators.keys())}')

        if parameter not in dependency.parameters:
            raise graph_scheduler.ConditionError(f'{dependency} has no {parameter} parameter')

        if (
            not isinstance(indices, graph_scheduler.TimeScale)
            and not isinstance(indices, collections.Iterable)
        ):
            indices = [indices]

        def func(dependency, parameter, threshold, comparator, indices, execution_id=None):
            param_value = self.get_parameter_value(execution_id)
            if isinstance(indices, graph_scheduler.TimeScale):
                param_value = param_value._get_by_time_scale(indices)
            elif indices is not None:
                for i in indices:
                    param_value = param_value[i]

            return comparison_operators[comparator](float(param_value), threshold)

        super().__init__(func, dependency, parameter, threshold, comparator, indices)

    def get_parameter_value(self, execution_id=None):
        return getattr(self.dependency.parameters, self.parameter).get(execution_id)
