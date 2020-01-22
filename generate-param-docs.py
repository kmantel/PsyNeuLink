import aenum
import enum
import inspect
import logging
import numpy as np
import os
import psyneulink as pnl
import re
import subprocess
import types
import typing

from psyneulink.core.globals.utilities import is_instance_or_subclass


def indent_str(string, num=0, indent_char=' ', indent_char_multiplier=4):
    return '{0}{1}'.format(
        '{0}'.format(
            ''.join([indent_char for i in range(indent_char_multiplier)])
        ).join(
            ['' for i in range(num + 1)]
        ),
        string
    )


def generate_param_desc_string(*args, indent=3):
    return '\n'.join([indent_str(x, indent) for x in args])


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(ch)

type_to_type_str = {
    int: '``int``',
    float: '``float``',
    None: '',
    str: '``str``',
    list: '``list``',
    bool: '``bool``',
    types.FunctionType: '``types.FunctionType``'

}

manual_descriptions = {
    pnl.Component: 'The `Parameters` that are associated with all `Components`',
    id(pnl.Component.parameters.is_finished_flag): (
        generate_param_desc_string(
            'internal parameter used by some Component types to track previous status of is_finished() method,',
            'or to set the status reported by the is_finished (see `is_finished <Component_Is_Finished>`',
        )
    ),
    id(pnl.ContrastiveHebbianMechanism.parameters.phase_convergence_threshold): (
        generate_param_desc_string(
            'internal parameter, used by is_converged;  assigned the value of the termination_threshold for',
            'the current `phase of execution <ContrastiveHebbian_Execution>`.',
        )
    ),
    id(pnl.InputPort.parameters.shadow_inputs): (
        generate_param_desc_string(
            'specifies whether InputPort shadows inputs of another InputPort;',
            'if not None, must be assigned another InputPort',
        )
    ),
    id(pnl.Port_Base.parameters.require_projection_in_composition): (
        generate_param_desc_string(
            'specifies whether the InputPort requires a projection when instantiated in a Composition;',
            'if so, but none exists, a warning is issued.',
        )
    ),
}

manual_parameter_types = {
    'random_state': '``numpy.random.RandomState``'
}


def replace_numpy_str(nparr):
    nparr = re.sub(r'\n *', ' ', repr(nparr))
    return 'numpy.{0}'.format(nparr)


def preprocess_params_list(params):
    # ensure variable and value are first and second, then sort
    fixed_strings = ['variable', 'value']
    fixed_params = []

    for i in range(len(fixed_strings)):
        for j in range(len(params)):
            if params[j].name == fixed_strings[i]:
                fixed_params.append(params[j])

    return fixed_params + sorted([x for x in params if x.name not in fixed_strings], key=lambda p: p.name)


def parse_type_to_type_string(typ):
    if is_instance_or_subclass(typ, pnl.Function):
        typ = '`{0}`'.format(pnl.Function.__name__)

    if is_instance_or_subclass(typ, (aenum.enum, enum.Enum)):
        typ = '`{0}`'.format(typ.__name__)

    if inspect.isclass(typ) and issubclass(typ, np.ndarray):
        typ = '``{0}.{1}``'.format(typ.__module__, typ.__name__)

    if isinstance(typ, list):
        typ = [parse_type_to_type_string(x) for x in typ]

    try:
        typ = type_to_type_str[typ]
    except KeyError:
        pass

    return typ


def parse_default_value_and_type_from_param(param):
    default_val_string, typ = parse_default_value_and_type(param.default_value)

    if isinstance(param._owner._owner, pnl.core.components.component.ComponentsMeta):
        try:
            # remove None out of optional
            typ = [
                x for x
                in typing.get_type_hints(param._owner._owner.__init__)[param.name].__args__
                # checking for NoneType not None
                if x is not type(None)
            ]
            if len(typ) == 1:
                typ = typ[0]
        except KeyError:
            pass
        except AttributeError:
            # tc.typecheck
            pass

    if typ is None:
        try:
            typ = manual_parameter_types[param.name]
            pass
        except KeyError:
            pass

    return default_val_string, parse_type_to_type_string(typ)


def parse_default_value_and_type(default_value):
    if default_value is None:
        default_val_string = 'None'
        typ = None
    elif isinstance(default_value, str):
        # camelcase to uppercase-underscore (std format of keywords)
        default_val_string = re.sub('([a-z0-9])([A-Z])', r'\1_\2', default_value).upper()
        default_val_string = '`{0}`'.format(default_val_string)
        typ = str
    elif isinstance(default_value, np.ndarray):
        default_val_string = '{0}'.format(replace_numpy_str(default_value))
        typ = type(default_value)
    elif isinstance(default_value, pnl.core.components.component.ComponentsMeta):
        default_val_string = '`{0}`'.format(default_value.__name__)
        typ = default_value
    elif isinstance(default_value, pnl.core.components.functions.optimizationfunctions.SampleIterator):
        default_val_string = '`{0}`'.format(default_value.__class__.__name__)
        typ = default_value.__class__
    # elif isinstance(default_value, types.MethodType):
    #     default_val_string = default_val_string.__qualname__
    elif isinstance(default_value, types.BuiltinFunctionType):
        if default_value.__module__ == 'builtins':
            # just give standard type, like float or int
            default_val_string = f'{default_value.__name__}'
        else:
            # some builtin modules are internally "_module"
            # but are imported with "module"
            default_val_string = f"{default_value.__module__.lstrip('_')}.{default_value.__name__}"
        default_val_string = f'``{default_val_string}``'
        typ = types.FunctionType
    elif isinstance(default_value, types.FunctionType):
        if '<lambda>' == default_value.__name__:
            default_val_string = re.sub('.*Parameter\\((.*?):(.*?)[,\\)].*', r'\1:\2', inspect.getsource(default_value)).strip()
        else:
            default_val_string = default_value.__qualname__
            try:
                # if the first portion of the function is a psyneulink name,
                # bracket it for doc reference
                split = default_val_string.split('.')
                try:
                    eval(f'pnl.{split[0]}')
                    default_val_string = f'`{split[0]}`.{split[1]}'
                except (NameError, AttributeError, SyntaxError):
                    pass
            except TypeError:
                pass

        typ = types.FunctionType
    elif isinstance(default_value, list):
        default_val_string = str([parse_default_value_and_type(x)[0] for x in default_value]).replace("'", "")
        typ = list
    else:
        if isinstance(default_value, pnl.Component):
            excluded_params = {'variable', 'value', 'previous_value', 'random_state'}
            default_val_string = '`{0}`({1})'.format(
                default_value.__class__.__name__,
                ', '.join([
                    '{0}={1}'.format(k, replace_numpy_str(v) if isinstance(v, np.ndarray) else v)
                    for (k, v) in sorted(default_value.defaults.values().items(), key=lambda x: x[0])
                    if (
                        v is not None
                        and k not in excluded_params
                        and 'param' not in k
                        and (isinstance(v, np.ndarray) or v != getattr(default_value.class_defaults, k))
                    )
                ])
            )
            default_val_string = re.sub('(.*)\\(\\)(.*)', r'\1\2', default_val_string)
            typ = default_value.__class__
            logger.info('FOUND INSTANCE, MAYBE NEEDS MANUAL INSPECTION')
        else:
            default_val_string = str(default_value)
            typ = type(default_value)

    return default_val_string, typ


def make_docstring_for_class(class_, module_fname):
    module_content = open(module_fname, 'r', encoding='utf-8').read()
    new_params = []

    for param in class_.parameters:
        if (
            param.user
            and param.name in class_.parameters.__class__.__dict__
            and (
                param.name not in class_.parameters._parent.__class__.__dict__
                or class_.parameters._parent.__class__.__dict__[param.name] is not class_.parameters.__class__.__dict__[param.name]
            )
        ):
            new_params.append(param)
        else:
            pass
            logger.debug('skipping nonspecial param {0}'.format(param.name))

    if len(new_params) == 0:
        return

    logger.debug('processing list {0}'.format([x.name for x in new_params]))
    new_params = preprocess_params_list(new_params)
    logger.debug('became {0}'.format([x.name for x in new_params]))

    result = ''
    result += '"""\n'

    # add optional description for classes
    try:
        result += indent_str('{0}\n\n'.format(manual_descriptions[class_]), 1)
    except KeyError:
        pass

    result += indent_str('Attributes\n', 1)
    result += indent_str('----------', 1)

    for param in new_params:
        result += '\n\n'
        result += indent_str(param.name + '\n', 2)
        basic_ref_str = '{1}.{0}'.format(param.name, class_.__name__)

        # attempt to find common format strings like Component_Variable
        manual_ref_str = basic_ref_str.replace('.', '_')
        manual_ref_str = re.sub(
            r'_(\w)',
            lambda match: '_' + match.group(1).upper(),
            manual_ref_str
        )
        sphinx_ref_str = f'.. _{manual_ref_str}:'

        if sphinx_ref_str in module_content:
            ref_str = manual_ref_str
        else:
            ref_str = basic_ref_str

        # attempt to find existent description
        # desc_str_pat = re.compile(rf'""".*{param.name}\n +[^(see)]', re.MULTILINE)
        # desc_str_pat = re.compile(rf'(.|\n)*{param.name}', re.MULTILINE)
        # match = desc_str_pat.findall(module_content)
        try:
            desc_str = manual_descriptions[id(param)]
        except KeyError:
            desc_str = indent_str('see `{0} <{1}>`'.format(param.name, ref_str), 3)

        result += desc_str
        result += '\n'

        default_val_string, typ = parse_default_value_and_type_from_param(param)

        # make default value
        result += '\n'
        result += indent_str(':default value: {0}'.format(default_val_string), 3)

        # make type
        result += '\n'
        result += indent_str(':type:{1}{0}'.format(typ, ' ' if len(str(typ)) > 0 else ''), 3)

        # make read only
        if param.read_only:
            result += '\n'
            result += indent_str(':read only: True', 3)

    result += '\n"""'

    return result


def print_all_pnl_docstrings():
    for item in pnl.__all__:
        item = eval('pnl.' + item)
        if isinstance(item, pnl.core.components.component.ComponentsMeta):
            if hasattr(item, 'parameters'):
                module_fname = item.__module__.replace('.', '/') + '.py'
                docstring = make_docstring_for_class(item, module_fname)
                if docstring is not None:
                    # assume that docstrings need to be on the 3rd level of indentation
                    # class A:
                    #     class Parameters
                    #         """docstring"""
                    # automatic calculation is better but this should be sufficient...
                    docstring = docstring.split('\n')
                    docstring = [indent_str(s, 2) if len(s) > 0 else '' for s in docstring]
                    docstring = '\n'.join(docstring)

                    logger.info('printing docstring for {0} residing in {1}'.format(item, module_fname))

                    dir_name = os.path.dirname(os.path.abspath(__file__))
                    subprocess.run([f'{dir_name}/setup-param-docs.sh', module_fname])
                    subprocess.run(['perl', f'{dir_name}/substitute-param-docs.pl', module_fname, item.__name__, docstring])

                    logger.info(docstring)
                    logger.info('================')


print_all_pnl_docstrings()

# search regex for selecting the docstring of a params class
# (?s)class Parameters[^\n]*?\n *""".*?"""

# in perl syntax:
# create docstrings for params that don't have them:
# perl -0777 -i.orig -pe 's;(class Parameters\([^\n]*?)\n( *)([^"\n]*)\n;\1\n\2"""\n\2"""\n\2\3\n;igs' psyneulink/core/components/component.py
# replace docstrings with some text:
# perl -0777 -i.orig -pe 's;(class Parameters\(.*?)\n( *)""".*?""";\1\n\2"""\n\2REPLACEME\n\2""";igs' psyneulink/core/components/component.py
