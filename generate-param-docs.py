import aenum
import enum
import inspect
import logging
import numpy as np
import psyneulink as pnl
import re
import subprocess
import types

from psyneulink.core.globals.utilities import is_instance_or_subclass

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(ch)

type_to_type_str = {
    int: 'int',
    float: 'float',
    None: '',
    str: 'str',
    list: 'list',
    bool: 'bool',
}

manual_descriptions = {
    pnl.Component: 'The `Parameters` that are associated with all `Components`',
}


def replace_numpy_str(nparr):
    nparr = re.sub(r'\n *', ' ', repr(nparr))
    return 'numpy.{0}'.format(nparr)


def indent_str(s, num=0, indent_char=' ', indent_char_multiplier=4):
    return '{0}{1}'.format(
        '{0}'.format(
            ''.join([indent_char for i in range(indent_char_multiplier)])
        ).join(
            ['' for i in range(num + 1)]
        ),
        s
    )


def preprocess_params_list(params):
    # ensure variable and value are first and second, then sort
    fixed_strings = ['variable', 'value']
    fixed_params = []

    for i in range(len(fixed_strings)):
        for j in range(len(params)):
            if params[j].name == fixed_strings[i]:
                fixed_params.append(params[j])

    return fixed_params + sorted([x for x in params if x.name not in fixed_strings], key=lambda p: p.name)


def parse_default_value_and_type_from_param(param):
    return parse_default_value_and_type(param.default_value)


def parse_default_value_and_type(default_value):
    if default_value is None:
        default_val_string = 'None'
        typ = ''
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

    if is_instance_or_subclass(typ, pnl.Function):
        typ = '`{0}`'.format(pnl.Function.__name__)

    if is_instance_or_subclass(typ, (aenum.enum, enum.Enum)):
        typ = '`{0}`'.format(typ.__name__)

    if inspect.isclass(typ) and issubclass(typ, np.ndarray):
        typ = '{0}.{1}'.format(typ.__module__, typ.__name__)

    try:
        typ = type_to_type_str[typ]
    except KeyError:
        pass

    return default_val_string, typ


def make_docstring_for_class(class_, module_fname):
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

        if sphinx_ref_str in open(module_fname, 'r', encoding='utf-8').read():
            ref_str = manual_ref_str
        else:
            ref_str = basic_ref_str
        # import code
        # code.interact(local=locals())
        result += indent_str('see `{0} <{1}>`\n'.format(param.name, ref_str), 3)

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

                    subprocess.run(['setup-param-docs.sh', module_fname])
                    subprocess.run(['perl', 'substitute-param-docs.pl', module_fname, item.__name__, docstring])

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
