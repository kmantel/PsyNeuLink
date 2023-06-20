# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# ***********************************************  Init ****************************************************************

"""
PsyNeuLink is a "block modeling system" for cognitive neuroscience.

Documentation is available at https://princetonuniversity.github.io/PsyNeuLink/

Example scripts are available at https://github.com/PrincetonUniversity/PsyNeuLink/tree/master/Scripts

If you have trouble installing PsyNeuLink, run into any bugs, or have suggestions for development,
please contact psyneulinkhelp@princeton.edu.
"""

import os
import logging as _logging

import numpy as _numpy
import pint as _pint

# pint requires a package-specific unit registry, and to use it as
# needed in psyneulink, it has to come before imports. This is the
# reason for skipping E402 below
_unit_registry = _pint.get_application_registry()
_pint.set_application_registry(_unit_registry)
_unit_registry.precision = 8  # TODO: remove when floating point issues resolved


# set up python logging for modification by users or other packages
# do before further imports so init messages are handled accordingly
# NOTE: logging is currently not used much if at all, if plan to use it
# more in the future, document for users how to enable/disable
def _get_default_log_handler():
    default_formatter = _logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler = _logging.StreamHandler()
    stream_handler.setFormatter(default_formatter)
    return stream_handler


def set_python_log_level(
    level: int = _logging.NOTSET, module: str = 'psyneulink', logger: _logging.Logger = None
):
    """
    Set python log level of **logger** (or logger for **module**) and
    all its handlers to **level**
    """
    if logger is None:
        logger = _logging.getLogger(module)
    logger.setLevel(level)

    if len(logger.handlers) == 0:
        logger.addHandler(_get_default_log_handler())

    for h in logger.handlers:
        h.setLevel(level)


def disable_python_logging(module: str = 'psyneulink', logger: _logging.Logger = None):
    """
    Disable python logging of **logger** (or logger for **module**)
    """
    set_python_log_level(_logging.CRITICAL + 1, module, logger)


_logging_env = os.environ.get('PNL_LOGGING')
if _logging_env:
    try:
        _logging_env = int(_logging_env)
    except (TypeError, ValueError):
        _logging_env = _logging_env.upper()

    if _logging_env == 0:
        disable_python_logging()
    else:
        if _logging_env == 1:
            level = _logging.DEBUG
        else:
            try:
                level = getattr(_logging, _logging_env)
            except AttributeError as e:
                raise ValueError(f'No logger level {_logging_env}') from e
            except TypeError:
                level = _logging_env

        set_python_log_level(level)


# starred imports to allow user imports from top level
from . import core  # noqa: E402
from . import library  # noqa: E402

from ._version import get_versions  # noqa: E402
from .core import *  # noqa: E402
from .library import *  # noqa: E402


_pnl_global_names = [
    'primary_registries', 'System', 'Process', '_unit_registry', 'logger', 'set_log_level'
]
# flag when run from pytest (see conftest.py)
_called_from_pytest = False

__all__ = list(_pnl_global_names)
__all__.extend(core.__all__)
__all__.extend(library.__all__)

# set __version__ based on versioneer
__version__ = get_versions()['version']
del get_versions

# suppress numpy overflow and underflow errors
_numpy.seterr(over='ignore', under='ignore')


primary_registries = [
    CompositionRegistry,
    DeferredInitRegistry,
    FunctionRegistry,
    MechanismRegistry,
    PathwayRegistry,
    PortRegistry,
    PreferenceSetRegistry,
    ProjectionRegistry,
]

for reg in primary_registries:
    def func(name, obj):
        if isinstance(obj, Component):
            obj._is_pnl_inherent = True

    process_registry_object_instances(reg, func)

def System(*args, **kwars):
    show_warning_sys_and_proc_warning()

def Process(*args, **kwars):
    show_warning_sys_and_proc_warning()

def show_warning_sys_and_proc_warning():
    raise ComponentError(f"'System' and 'Process' are no longer supported in PsyNeuLink; "
                         f"use 'Composition' and/or 'Pathway' instead")


del os
