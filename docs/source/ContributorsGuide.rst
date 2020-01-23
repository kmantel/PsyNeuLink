Contributors Guide
==================

* `Introduction`
* `File_Structure`
* `Environment_Setup`
* `Contribution_Checklist`
* `Components_Overview`
* `Component_Creation`
* `Compositions_Overview`
* `Scheduler`
* `Testing`
* `Documentation`
* `Example`

.. _Introduction:

Introduction
------------

Thank you for your interest in contributing to PsyNeuLink! This page is written and maintained by contributors to
PsyNeuLink. It provides helpful information for new contributors that complements the user documentation.

.. _File_Structure:

File Structure
--------------

In the PsyNeuLink repo, there are many files. The following folders and files are the most relevant:

- *docs*:  directory that contains the documentation files, including this Contributors Guide

  * *source*: directory that contains the Sphinx files used to generate the HTML documentation
  * *build*: directory that contains the generated HTML documentation, which is generated using Sphinx commands

- *Scripts*:  directory that contains sample PsyNeuLink scripts. Not all of these scripts are actively maintained, and
  some may be outdated

- *tests*: directory that contains test code used by pytests, and is actively maintained

- *CONVENTIONS.md*: file that describes coding conventions that contributors must follow, such as documentation style
  and variable naming

- *psyneulink*: directory that contains the source code for PsyNeuLink

  * *core*: directory that contains the core objects of psyneulink
  * *library*: directory that contains user-contributed extensions to psyneulink and other non-core objects

.. _Environment_Setup:

Environment Setup and Installaion
---------------------------------

PsyNeuLink currently supports Python 3.6+, and we aim to support all future releases of Python.
First install Python and pip on your machine, if not installed already.
We suggest `anaconda <https://www.anaconda.com/>`_ or `pyenv <https://github.com/pyenv/pyenv>`_.
Next, clone the PsyNeuLink git repository.
Finally, navigate to the PsyNeuLink folder and install development dependencies::

    pip install -e .[dev]

If necessary, use `pip3` instead of `pip`.

PsyNeuLink uses `pytest <https://docs.pytest.org/en/latest/index.html>`_ to run its tests.
To build documentation, we use `Sphinx <https://www.sphinx-doc.org/en/master/usage/installation.html>`_.
To contribute, make a branch off of the ``devel`` branch.
Make a pull request to ``devel`` once your changes are complete.
``devel`` is periodically merged into the ``master`` branch, which is the branch most users use and is installed with a standard pip install.

.. _Contribution_Checklist:

Contribution Checklist
----------------------

This is the general workflow for contributing to PsyNeuLink:

* Using git, create a branch off of the ``devel`` branch.
* Make your changes to the code. Ideally, notify the PsyNeuLink team in advance of what you intend to do, so that
  they can provide you with relevant tips in advance.

  * While writing code on your branch, be sure to keep pulling from `devel` from time to time! Since PsyNeuLink is
    actively being developed, substantial changes may have been made to the code base on ``devel`` while you were
    working on your branch;  getting too far behind these may make it difficult for you to merge your branch when you
    are ready.
  * Be sure to write documentation for your new classes or functions, in the style of other PsyNeuLink classes.

* Once you've completed the changes and/or additions on your branch, add tests that check that these
  works as expected. This helps ensure that other developers don't accidentally break your code when making their own
  changes!
* Once your changes are complete and working, run the `pytest <https://docs.pytest.org/en/latest/index.html>`_ tests
  and make sure all tests pass. If you encounter unexpected test failures, please notify the PsyNeuLink team.
* Once all tests pass, submit a pull request to the PsyNeuLink devel branch! The PsyNeuLink team will then review your
  changes and accept the pull request if they sastify the requirements described above.

.. _Components_Overview:

Components Overview
-------------------

Most PsyNeuLink objects are `Components <Component>`. All `Functions <Function>`, `Mechanisms <Mechanism>`,
`Projections <Projection>`, and `Ports <Port>` are subclasses of Component. These subclasses use and override many
functions from the Component class, so they are initialized and executed in similar ways.

The subclasses of Component should override Component's functions to implement their own functionality.
However, function overrides must call the overridden function using `super()`, while passing the same arguments.
For example, to instantiate a Projection's receiver after instantiating its function,
the Projection_Base class overrides the `_instantiate_attributes_after_function` as follows::

    class Projection_Base(Projection):
        def _instantiate_attributes_after_function(self, context=None):
            self._instantiate_receiver(context=context)
            super()._instantiate_attributes_after_function(context=context)

If you wish to modify the behavior of a `Mechanism <Mechanism>`, `Projection <Projection>`, or `Port <Port>` in
PsyNeuLink, most likely you will *not* need to create an entirely new subclass.  Usually, this can be
accomplished by assigning a custom function to an existing class, either by assigning it an instance of a
`UserDefinedFunction` (in the case of simple computations), or by creating a new subclass of `Function` (for more
complex computations).  A new subclass of `Mechanism <Mechanism>`, `Projection <Projection>`, or `Port <Port>`
should be created only if the desired behavior requires a significant deviation from the usual execution pattern.  If
that is the case, be sure to file an issue in the repo outlining your needs and your plan for addressing them, so that
members of the team can advise you if there is an easier way of approaching the problem.

Parameters vs. Attributes
^^^^^^^^^^^^^^^^^^^^^^^^^

Any attribute of a Component that you create should generally implemented as a `Parameter` rather than a simple
python attribute.  This ensures that it will be:

  * threadsafe and correct in all contexts (see below);
  * accessible to the user with standard PsyNeuLink notations and/or procedures;
  * accessible for `modulation <ModulatorySignal_Modulation>` by PsyNeuLink's Modulatory Components;
  * properly represented in a `json` export of a script that uses your Component.

See the `developer documentation for Parameters <Parameter_Developers>` for additional information.

Context and Statefulness
^^^^^^^^^^^^^^^^^^^^^^^^

Any modifications you make to a `Component` must be aware of its `Context` object, and manage it appropriately, or
the Component is likely to produce incorrect behaviors or crash. A `Context` object stores information about the
current state of execution of the Component to which it belongs, and must be passed through most PsyNeuLink methods
and functions called on that Component. Also, `Parameter` values must always be set and retrieved using a `Context`
object (see `here <Parameter_Use>` for additional information).

Default contexts are specified for a Component when it is executed within `Composition.run`.  When using
non-default contexts outside of Compositions, `_initialize_from_context` must be called manually. The below code will
fail, because ``m`` has no parameter values for ``'some custom context'``::

    m = pnl.ProcessingMechanism()
    m.execute(1, context='some custom context')

To fix this, ``'some custom context'`` must be initialized beforehand, as follows::

    m._initialize_from_context(context=Context(execution_id='some custom context'))


.. _Component_Creation:

Creating a Custom Subclass of Component
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _Component_Initialization:

*Initialization*
~~~~~~~~~~~~~~~~

*Parameter specification*

The constructor (``__init__`` method) of new sublcass should include an explicit argument for each `Parameter` that
is introduced in the subclass (i.e., that is not defined in the parent class) and/or any that needs preprocessing in
the constructor before being passed to the parent class for completion of initialization. Any others are implicitly
through the ``__init__`` hierarchy in the ``**kwargs`` argument.

Default/initial values for
all these parameters should be set in the `Parameters` class, instead of the python standard default argument value,
which should be set to ``None``. This is to ensure that the `_user_specified <Parameter._user_specified>` attribute is
set correctly, which is used to indicate whether the value for a Parameter was explicitly given by the user or its
default value was assigned. e.g.::

    >>> import psyneulink as pnl
    >>> f = pnl.Linear(slope=2)
    >>> f.parameters.slope._user_specified
    True
    >>> f.parameters.intercept._user_specified
    False

*Initialization sequence*

Broadly, the sequence of events for initialization of a `Component` are as follows:

#. Call ``__init__`` methods in hierarchical order (``__init__``, ``super().__init__()``, etc.).
#. Set Parameter default values based on input and `class defaults <Component.class_defaults>`
   (``_initialize_parameters``).
#. Set default `variable <Component.variable>` based on input (``default_variable`` and any other Parameters on which
   it depends) and class defaults (``_handle_default_variable``).
#. Call ``_instantiate_attributes_before_function`` hook.
#. Construct, copy, or assign function (``_instantiate_function``).
#. Execute once to produce a default `value <Component.value>` (``_instantiate_value``).
#. Call ``_instantiate_attributes_after_function`` hook.

.. [## I THINK IT WOULD BE GOOD TO HAVE SLIGHTLY MORE INFORMATION ABOUT WHY EACH OF THESE METHODS IS THERE AND WHAT
   THEY (CAN BE USED TO) DO.  WHILE I TOTALLY AGREE THIS DOCUMENT SHOULD BE AS CONCISE AS POSSIBLE, I ALSO THINK IT
   WILL BE HELPFUL TO HAVE, IN ONE PLACE, THE RATIONALE FOR THE OVERALL ARCHITECTURE / PROCESS FLOW].

*Execution*
~~~~~~~~~~~

Components (excluding Compositions) run the following steps during `execution <Component_Execution>`.

1. Call `_parse_function_variable` on the input `variable <Component.variable>`, which reformats `variable` for use with the function, if necessary
2. Call `function <Component.function>` on the result of 1., which does the primary computation for the Component

`Mechanisms <Mechanism>` add a few extra steps:

1. If no variable is passed in, call `_update_input_ports`,
to update values of the `input_ports <Mechanism.input_ports>` based on their functions, and use these as the input `variable <Mechanism.variable>` (if no variable was manually specified)
2. Call `_update_parameter_ports`, which updates the values of any `modulated parameters <ModulatorySignal_Modulation>` to be used in the Mechanism's function
3. Call `_parse_function_variable` on the input `variable`
4. Call `function <Component.function>` on the result of 3.
5. Call `_update_output_ports`, which updates the values of the `output_ports <Mechanism.output_ports>` based on their functions. These values are passed on to other Mechanisms as applicable
6. If `execute_until_finished <Component_Execute_Until_Finished>` is `True`, repeat steps 1-5 until one of the following:

   a. `is_finished <Component.is_finished>` returns ``True``
   b. `num_executions_before_finished <Component.num_executions_before_finished>` is greater than or equal to `max_executions_before_finished <Component.max_executions_before_finished>`

.. [## AGAIN, I THINK IT WOULD BE GOOD TO HAVE SLIGHTLY MORE INFORMATION ABOUT WHY EACH OF THESE METHODS IS THERE AND
   WHAT THEY (CAN BE USED TO) DO]

.. _Compositions_Overview:

Compositions Overview
---------------------

Execution
^^^^^^^^^

The execution of a `Composition` is handled by `run <Composition.run>`, `execute <Composition.execute>` as a helper
to `run`, and `evaluate <Composition.evaluate>` that is used to simulate the execution of a Composition when it is
assigned as the `agent_rep <OptimizationControlMechanism.agent_rep>` of an `OptimizationControlMechanism`. One call to `run` corresponds to one `RUN <TimeScale.RUN>` of time, and follows these steps:

1. `reinitialize <Component.reinitialize>` for nodes in `reinitialize_values`
2. `analyze_graph`
3. initialize contexts

    a. `assign_execution_ids`
    b. `_initialize_from_context`

4. loop over trials

    a. `call_before_trial`
    b. Check whether `RUN` `termination conditions <Scheduler_Termination_Conditions>` for the Composition have been met, and if so, go to 5.
    c. Process inputs to be used for each `TRIAL`, see `Composition_Run_Inputs`
    d. Reinitialize any nodes whose `reinitialize_when` Condition is method
    e. Run a single trial by calling `execute <Contributors_Composition_Execute>`
    f. `call_after_trial`

5. Delete stored `simulation <>` results and data if `retain_old_simulation_data` is ``False``, because these structures can grow expensively large
6. Add the results of each `TRIAL <TimeStep.TRIAL>`, in order, to `results <Composition.results>`.

.. _Contributors_Composition_Execute:

`execute <Composition.execute>` completes a single `TRIAL` with the following steps:

1. Initialize the execution Context as in `Execution` step 3. above (but not if `execute` is called through `run`
2. Assign inputs to
3.

. `Execute the learning phase <Composition_Learning_Execution>` if applicable

.. _Scheduler:

Scheduler
---------

Customizing scheduling can almost always be handled by adding `Condition`\s. `Condition`\s that require
no stored state can be created ad-hoc, using just an instance of
`Condition <psyneulink.core.scheduling.condition.Condition>`, `While`, or `WhileNot`.
If a Condition is need that requires stored state, then to implement a subclass you should create a function that
returns `True` if the condition is satisfied, and `False` otherwise, and assign it to the `func <Condition.func>`
attribute of the `Condition`. Any ``*args`` and ``**kwargs`` passed in to `Condition.__init__ psyneulink.core.scheduling.condition.Condition>` will be given, unchanged, to each call of `func <Condition.func>`, along with an
``execution_id``.

.. note::

    Your stored state must be independent for each ``context``/``execution_id``

.. _Testing:

Testing
-------

PsyNeuLink uses `pytest <https://docs.pytest.org/en/latest/>`_ and a test suite in the ``tests`` directory.
When contributing, you should include tests with your submission. You may find it helpful to create
tests for your contribution before writing it, to help you achieve your desired behavior. Code and documentation
style is enforced by the python modules ``pytest-pycodestyle`` and ``pytest-pydocstyle``.

To run all the tests that must pass for your contribution to be accepted, simply run ``pytest`` in the `PsyNeuLink`
directory.

.. _Documentation:

Documentation
-------------

Documentation is done in docstrings for the PsyNeuLink objects using the Sphinx library. Documentation for the
`master` and `devel` branches can be found `here <https://princetonuniversity.github.io/PsyNeuLink/>`_ and
`here <https://princetonuniversity.github.io/PsyNeuLink/branch/devel/index.html>`_, respectively.

To understand Sphinx syntax, start
`here <http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_ .
When creating and/or editing documentation, you should generate Sphinx documentation in order to preview your changes
before publishing to `devel`. To generate Sphinx documentation from your local branch, run `make html` in Terminal
while in the `docs` folder. The resulting HTML should be in your `docs/build` folder. (Do not commit these built HTML
files to Github. They are simply for your local testing/preview purposes.)

.. _Example:

Example
-------

Here, we will create a custom Function, ``RandomIntegrator`` that uses stored state and randomness. ``RandomIntegrator`` stores two values, `previous_value` (used in many PNL Functions) and `previous_value_2` (a second value chosen just for this example). ``RandomIntegrator`` chooses one randomly at execution time, increments it by the input `variable <Component>`, and returns the result.

1. Inherit from a relevant PsyNeuLink Component; use `IntegratorFunction` so that we have access to
   its `previous_value <IntegratorFunction.previous_value>` and `rate <IntegratorFunction.rate>` Parameters::

        class RandomIntegrator(IntegratorFunction):

2. Create a subclass of `Parameters` from the one defined for `IntegratorFunction` that adds attributes we will need::

        class Parameters(IntegratorFunction.Parameters):

            random_state = Parameter(None, pnl_internal=True)
            previous_value_2 = Parameter(np.array([1000]), pnl_internal=True)

.. [JDC: NOT SURE I FULLY UNDERSTAND THE RATIONALE FOR previous_value_2 AS EXPLAINED BELOW]
.. [KDM: Added above. It's meant to be arbitrary and somewhat pointless (otherwise, wouldn't we want to include this integrator as a built-in?).]

``random_state`` will be used to generate random numbers statefully and independently.
``previous_value_2`` will be used in our function, and has its default value set arbitrarily to 10, to distinguish it
from `previous_value <IntegratorFunction.previous_value>` which is created on `IntegratorFunction.Parameters` and so
does not need to be overridden here. We set the attribute `pnl_internal` to ``True`` on each of these Parameters
for use with the `JSON/OpenNeuro collaboration <json>`, to indicate that they are not relevant to modeling platforms
other than PsyNeuLink.

3. Create an ``__init__`` method::

        def __init__(
            self,
            seed=None,
            previous_value_2=None,
            **kwargs
        ):
            if seed is None:
                seed = get_global_seed()

            super().__init__(
                previous_value_2=previous_value_2,
                random_state=np.random.RandomState([seed]),
                **kwargs
            )

Note that the default value for ``previous_value_2`` is ``None`` (`see above <Component_Initialization>`).
Any other Parameters will be handled through `**kwargs`. ``seed`` is not simply the standard time-based seed for testing and replication purposes. See `get_global_seed`.

.. [JDC: WHAT ABOUT SEED?  SHOULDN'T THAT BE MENTIONED EARLIER OR HERE?]

.. [JDC:  CHECK FOLLOWING EDITTED STATEMENT FOR ACCURACY]

4. Write a ``_function`` method (this will be automatically wrapped and accessible as the Component's `function <Component_Function>` method)::

        def _function(
            self,
            variable=None,  # the main input
            context=None,
            params=None,    # future use, runtime_params
        ):
            rate = self.get_current_function_param('rate', context)
            if self.parameters.random_state._get(context).choice([1, 2]) == 1:
                new_value = self.parameters.previous_value._get(context) + rate * variable
                self.parameters.previous_value._set(new_value, context)
            else:
                new_value = self.parameters.previous_value_2._get(context) + rate * variable
                self.parameters.previous_value_2._set(new_value, context)

            return self.convert_output_type(new_value)

When an instance of ``RandomIntegrator`` is executed, and its `function <Component.function>` method is called, it
chooses one of its previous values, adds the product of `rate` and `variable` to it, stores the result back into the appropriate previous value, and returns the result.

.. [JDC: WHERE IN THE SOURCE CODE IS THE INFORMATION BELOW EXPLAINED... IN THE DOCSTRING FOR
   get_current_function_param AND/OR _get?  IF NOT, THEN NEED TO REFERENCE WHEREVER IT IS EXPLAINED].
   ALSO, WHY ISN'T get_current_function_param UNDERSCORED?  IS IT MEANT TO BE USER (NOT JUST CONTRIBUTOR)
   ACCESSIBLE?

We use `get_current_function_param` instead of just `_get` for ``rate``, because it is a `modulable Parameter <Parameter.modulable>`, meaning it has an associated `ParameterPort` on its owner Mechanism, ``RandomIntegrator``.
This ensures that if ``rate`` is subject to `modulation <ModulatorySignal_Modulation>`, its modulated value is
returned;  otherwise, its base value would be used, which is equivalent to value returned by `_get`.  In contrast,
neither `previous_value` nor `previous_value_2` are not modulable, and so we can simply use `_get` for them.

We call `convert_output_type` before returning as a general pattern on Functions with simple output (see
`Function_Output_Type_Conversion` for additional information).

Below is the fully implemented class, ready to be included in PsyNeuLink::

    import numpy as np
    from psyneulink import IntegratorFunction, Parameter, get_global_seed


    class RandomIntegrator(IntegratorFunction):

        class Parameters(IntegratorFunction.Parameters):

            random_state = Parameter(None, pnl_internal=True)
            previous_value_2 = Parameter(np.array([1000]), pnl_internal=True)

        def __init__(
            self,
            seed=None,
            previous_value_2=None,
            **kwargs
        ):
            if seed is None:
                seed = get_global_seed()

            super().__init__(
                previous_value_2=previous_value_2,
                random_state=np.random.RandomState([seed]),
                **kwargs
            )

        def _function(
            self,
            variable=None,  # the main input
            context=None,
            params=None,    # future use, runtime_params
        ):
            rate = self.get_current_function_param('rate', context)
            if self.parameters.random_state._get(context).choice([1, 2]) == 1:
                new_value = self.parameters.previous_value._get(context) + rate * variable
                self.parameters.previous_value._set(new_value, context)
            else:
                new_value = self.parameters.previous_value_2._get(context) + rate * variable
                self.parameters.previous_value_2._set(new_value, context)

            return self.convert_output_type(new_value)
