import sys

dot_template = 'from . import {0}'
star_template = 'from .{0} import *'
all_first_template = '__all__ = list({0}.__all__)'
all_rest_template = '__all__.extend({0}.__all__)'


def _create_strings_from_template(*args, template):
    return [template.format(arg) for arg in args]


def create_import_strings(*args):
    if len(args) < 1:
        return ''

    sorted_args = sorted([x for x in args])
    final = ''

    final += '\n'.join(_create_strings_from_template(*sorted_args, template=dot_template))
    final += '\n\n'
    final += '\n'.join(_create_strings_from_template(*sorted_args, template=star_template))
    final += '\n\n'
    final += all_first_template.format(sorted_args[0])
    final += '\n' if len(sorted_args) > 1 else ''
    final += '\n'.join(_create_strings_from_template(*sorted_args[1:], template=all_rest_template))

    return final


if __name__ == '__main__':
    print(create_import_strings(*sys.argv[1:]))
