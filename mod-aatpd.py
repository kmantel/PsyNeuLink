import psyneulink
import glob
import re
import pprint
import types

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
