import psyneulink

import glob
import re
import os
import pprint
import types


visited = set()


def process_module(module):
    if module in visited:
        return

    has_submodules = False
    visited.add(module)

    for (name, submodule) in module.__dict__.items():
        if isinstance(submodule, types.ModuleType):
            submodule_fname = str(submodule).replace('.', '/') + '.py'
            try:
                open(submodule_fname, 'r', encoding='utf-8')
            except FileNotFoundError:
                print('skipping', submodule)
                continue

            process_module(submodule)
            has_submodules = True

    if not has_submodules:
        print(module)

        module_fname = str(module).replace('.', '/') + '.py'

        try:
            with open(module_fname, 'r', encoding='utf-8') as fp:
                for pnl_class in module.__all__:
                    if isinstance(pnl_class, psyneulink.core.components.component.ComponentsMeta):
                        file = fp.read()
                        res = re.findall(
                            fr'({pnl_class.__name__}).*?\n( )*(params = self._assign_args_to_param_dicts\(.*?\))',
                            file,
                            flags=re.MULTILINE | re.DOTALL
                        )

                        if res and len(res):
                            pprint.pprint(res)
        except FileNotFoundError:
            return


for filepath in glob.glob(f'**/*.py', recursive=True):
    module_name = filepath.replace('/', '.').replace('.py', '')
    try:
        module = eval(module_name)
        if (
            not isinstance(module, types.ModuleType)
            or not hasattr(module, '__all__')
        ):
            continue
        else:
            print('found one', module_name)

    except (AttributeError, NameError, SyntaxError):
        continue

    with open(filepath, 'r', encoding='utf-8') as fp:
        for pnl_class in module.__all__:
            pnl_class = eval(f'psyneulink.{pnl_class}')
            if isinstance(pnl_class, psyneulink.core.components.component.ComponentsMeta):
                file = fp.read()
                res = re.findall(
                    fr'({pnl_class.__name__}).*?\n( )*(params = self._assign_args_to_param_dicts\(.*?\))',
                    file,
                    flags=re.MULTILINE | re.DOTALL
                )

                if res and len(res):
                    pprint.pprint(res)

#     with open(filepath, 'r', encoding='utf-8') as fp:
# for item in pnl.__all__:


# for (name, module) in psyneulink.__dict__.items():
#     if isinstance(module, types.ModuleType):
#         process_module(module)


# for item in psyneulink.core.components.functions.learningfunctions.__all__:
#     item = eval('psyneulink.' + item)
#     # item = pnl.BayesGLM

#     if isinstance(item, psyneulink.core.components.component.ComponentsMeta):
#         module = eval(item.__module__)
#         module_fname = item.__module__.replace('.', '/') + '.py'

#         with open(module_fname, 'r', encoding='utf-8') as fp:
#             for pnl_class in module.__all__:
#                 if isinstance(pnl_class, psyneulink.core.components.component.ComponentsMeta):
#                     file = fp.read()
#                     res = re.findall(
#                         fr'({pnl_class.__name__}).*?\n( )*(params = self._assign_args_to_param_dicts\(.*?\))',
#                         file,
#                         flags=re.MULTILINE | re.DOTALL
#                     )

#                     if res and len(res):
#                         pprint.pprint(res)
