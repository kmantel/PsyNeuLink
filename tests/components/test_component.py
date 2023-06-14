import numpy as np
import psyneulink as pnl
import pytest


def nest_dictionary(elem, keys=NotImplemented):
    """
    Args:
        elem
        keys (object, iterable, optional)

    Returns:
        dict: **elem** if **keys** is NotImplemented, or a dictionary
        containing **elem** nested by **keys**
    """
    if keys is NotImplemented:
        return elem

    if isinstance(keys, str):
        keys = [keys]

    try:
        iter(keys)
    except TypeError:
        keys = [keys]

    res = elem
    for k in reversed(keys):
        res = {k: res}
    return res


class TestComponent:

    def test_detection_of_legal_arg_in_kwargs(self):
        assert isinstance(pnl.ProcessingMechanism().reset_stateful_function_when, pnl.Never)
        assert isinstance(pnl.ProcessingMechanism(reset_stateful_function_when=pnl.AtTrialStart()).reset_stateful_function_when,
                          pnl.AtTrialStart)

    def test_detection_of_illegal_arg_in_kwargs(self):
        with pytest.raises(pnl.ComponentError) as error_text:
            pnl.ProcessingMechanism(flim_flam=1)
        assert "Unrecognized argument in constructor for ProcessingMechanism-0 (type: ProcessingMechanism): 'flim_flam'" in str(error_text)

    def test_detection_of_illegal_args_in_kwargs(self):
        with pytest.raises(pnl.ComponentError) as error_text:
            pnl.ProcessingMechanism(name='MY_MECH', flim_flam=1, grumblabble=2)
        assert "Unrecognized arguments in constructor for MY_MECH (type: ProcessingMechanism): 'flim_flam, grumblabble'" in str(error_text)

    def test_component_execution_counts_for_standalone_mechanism(self):

        T = pnl.TransferMechanism()

        T.execute()
        assert T.execution_count == 1
        assert T.input_port.execution_count == 1 # incremented by Mechanism.get_variable_from_input()

        # skipped (0 executions) because execution is bypassed when no afferents, and
        # function._is_identity is satisfied (here, Linear function with slope 0 and intercept 1)
        # This holds true for each below
        assert T.parameter_ports[pnl.SLOPE].execution_count == 0
        assert T.output_port.execution_count == 0

        T.execute()
        assert T.execution_count == 2
        assert T.input_port.execution_count == 2
        assert T.parameter_ports[pnl.SLOPE].execution_count == 0
        assert T.output_port.execution_count == 0

        T.execute()
        assert T.execution_count == 3
        assert T.input_port.execution_count == 3
        assert T.parameter_ports[pnl.SLOPE].execution_count == 0
        assert T.output_port.execution_count == 0

    def test_component_execution_counts_for_mechanisms_in_composition(self):

        T1 = pnl.TransferMechanism()
        T2 = pnl.TransferMechanism()
        c = pnl.Composition()
        c.add_node(T1)
        c.add_node(T2)
        c.add_projection(sender=T1, receiver=T2)

        input_dict = {T1:[[0]]}

        c.run(input_dict)
        assert T2.execution_count == 1
        assert T2.input_port.execution_count == 1
        assert T2.parameter_ports[pnl.SLOPE].execution_count == 0
        assert T2.output_port.execution_count == 0

        c.run(input_dict)
        assert T2.execution_count == 2
        assert T2.input_port.execution_count == 2
        assert T2.parameter_ports[pnl.SLOPE].execution_count == 0
        assert T2.output_port.execution_count == 0

        c.run(input_dict)
        assert T2.execution_count == 3
        assert T2.input_port.execution_count == 3
        assert T2.parameter_ports[pnl.SLOPE].execution_count == 0
        assert T2.output_port.execution_count == 0

    def test__set_all_parameter_properties_recursively(self):
        A = pnl.ProcessingMechanism(name='A')
        A._set_all_parameter_properties_recursively(history_max_length=0)

        for c in A._dependent_components:
            for param in c.parameters:
                assert param.history_max_length == 0

    @pytest.mark.parametrize(
        'component_type', [
            pnl.ProcessingMechanism,
            pnl.TransferMechanism,
            pnl.Linear,
            pnl.DDM
        ]
    )
    def test_execute_manual_context(self, component_type):
        c = component_type()
        default_result = c.execute(5)

        assert pnl.safe_equals(c.execute(5, context='new'), default_result)


class TestConstructorArguments:
    @pytest.mark.parametrize(
        'cls_',
        [
            pnl.ProcessingMechanism,
            pytest.param(
                pnl.IntegratorMechanism,
                marks=pytest.mark.xfail(reason='size currently unsupported at all on IntegratorMechanism')
            )
        ]
    )
    @pytest.mark.parametrize('params_dict_entry', [NotImplemented, 'params'])
    def test_size(self, cls_, params_dict_entry):
        c = cls_(**nest_dictionary({'size': 5}, params_dict_entry))
        assert len(c.defaults.variable[-1]) == 5

    @pytest.mark.parametrize(
        'cls_, function_params, expected_values',
        [
            (pnl.ProcessingMechanism, {'slope': 2}, NotImplemented),
        ]
    )
    @pytest.mark.parametrize('params_dict_entry', [NotImplemented, 'params'])
    def test_function_params(self, cls_, function_params, expected_values, params_dict_entry):
        m = cls_(**nest_dictionary({'function_params': function_params}, params_dict_entry))

        if expected_values is NotImplemented:
            expected_values = function_params

        for k, v in expected_values.items():
            np.testing.assert_array_equal(getattr(m.function.defaults, k), v)

    @pytest.mark.parametrize(
        'cls_, param_name, argument_name, param_value',
        [
            (pnl.TransferMechanism, 'variable', 'default_variable', [[10]]),
        ]
    )
    @pytest.mark.parametrize('params_dict_entry', [NotImplemented, 'params'])
    def test_valid_argument(self, cls_, param_name, argument_name, param_value, params_dict_entry):
        obj = cls_(**nest_dictionary({argument_name: param_value}, params_dict_entry))
        np.testing.assert_array_equal(getattr(obj.defaults, param_name), param_value)
