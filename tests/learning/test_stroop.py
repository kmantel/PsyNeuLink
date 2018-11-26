import numpy as np
import psyneulink as pnl

from psyneulink.core.components.functions.learningfunctions import BayesGLM
from psyneulink.core.components.functions.optimizationfunctions import GridSearch
from psyneulink.core.components.functions.transferfunctions import Exponential, Linear, Logistic
from psyneulink.core.components.functions.userdefinedfunction import UserDefinedFunction
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.components.process import Process
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.components.system import System
from psyneulink.core.globals.keywords import FULL_CONNECTIVITY_MATRIX, LEARNING, LEARNING_PROJECTION
from psyneulink.core.globals.preferences.componentpreferenceset import REPORT_OUTPUT_PREF, VERBOSE_PREF
from psyneulink.library.components.mechanisms.processing.objective.comparatormechanism import MSE


class TestStroop:

    def test_stroop_model(self):
        process_prefs = {
            REPORT_OUTPUT_PREF: False,
            VERBOSE_PREF: False
        }

        # system_prefs = {
        #     REPORT_OUTPUT_PREF: True,
        #     VERBOSE_PREF: False
        # }

        colors = TransferMechanism(
            size=2,
            function=Linear,
            name="Colors",
        )

        words = TransferMechanism(
            default_variable=[0, 0],
            size=2,
            function=Linear,
            name="Words",
        )

        response = TransferMechanism(
            default_variable=[0, 0],
            function=Logistic,
            name="Response",
        )

        color_naming_process = Process(
            default_variable=[1, 2.5],
            pathway=[colors, FULL_CONNECTIVITY_MATRIX, response],
            learning=LEARNING_PROJECTION,
            target=[0, 1],
            name='Color Naming',
            prefs=process_prefs,
        )

        word_reading_process = Process(
            default_variable=[.5, 3],
            pathway=[words, FULL_CONNECTIVITY_MATRIX, response],
            name='Word Reading',
            learning=LEARNING_PROJECTION,
            target=[1, 0],
            prefs=process_prefs,
        )

        # s = System(
        #     processes=[color_naming_process, word_reading_process],
        #     name='Stroop Model',
        #     targets=[0, 0],
        #     prefs=system_prefs,
        # )

        # stim_dict = {
        #     colors: [
        #         [1,0],
        #         [0,1]
        #     ],
        #     words: [
        #         [0,1],
        #         [1,0]
        #     ]
        # }
        # target_dict = {
        #     response: [
        #         [1,0],
        #         [0,1]
        #     ]
        # }

        # results = s.run(
        #     num_trials=10,
        #     inputs=stim_dict,
        #     targets=target_dict,
        # )
        expected_color_results = [
            np.array([0.88079708, 0.88079708]),
            np.array([0.85997037, 0.88340023]),
            np.array([0.83312329, 0.88585176]),
            np.array([0.79839127, 0.88816536]),
            np.array([0.75384913, 0.89035312]),
            np.array([0.69835531, 0.89242571]),
            np.array([0.63303376, 0.89439259]),
            np.array([0.56245802, 0.8962622 ]),
            np.array([0.49357614, 0.89804208]),
            np.array([0.43230715, 0.89973899]),
        ]

        expected_word_results = [
            np.array([0.88079708, 0.88079708]),
            np.array([0.88340023, 0.85997037]),
            np.array([0.88585176, 0.83312329]),
            np.array([0.88816536, 0.79839127]),
            np.array([0.89035312, 0.75384913]),
            np.array([0.89242571, 0.69835531]),
            np.array([0.89439259, 0.63303376]),
            np.array([0.8962622, 0.56245802]),
            np.array([0.89804208, 0.49357614]),
            np.array([0.89973899, 0.43230715]),
        ]

        for i in range(10):
            cr = color_naming_process.execute(input=[1, 1], target=[0, 1])
            wr = word_reading_process.execute(input=[1, 1], target=[1, 0])

            np.testing.assert_allclose(cr, expected_color_results[i], atol=1e-08, err_msg='Failed on expected_color_results[{0}]'.format(i))
            np.testing.assert_allclose(wr, expected_word_results[i], atol=1e-08, err_msg='Failed on expected_word_results[{0}]'.format(i))

    def test_stroop_model_learning(self):
        process_prefs = {
            REPORT_OUTPUT_PREF: True,
            VERBOSE_PREF: False,
        }
        system_prefs = {
            REPORT_OUTPUT_PREF: True,
            VERBOSE_PREF: False,
        }

        colors = TransferMechanism(
            default_variable=[0, 0],
            function=Linear,
            name="Colors",
        )
        words = TransferMechanism(
            default_variable=[0, 0],
            function=Linear,
            name="Words",
        )
        hidden = TransferMechanism(
            default_variable=[0, 0],
            function=Logistic,
            name="Hidden",
        )
        response = TransferMechanism(
            default_variable=[0, 0],
            function=Logistic(),
            name="Response",
        )
        TransferMechanism(
            default_variable=[0, 0],
            function=Logistic,
            name="Output",
        )

        CH_Weights_matrix = np.arange(4).reshape((2, 2))
        WH_Weights_matrix = np.arange(4).reshape((2, 2))
        HO_Weights_matrix = np.arange(4).reshape((2, 2))

        CH_Weights = MappingProjection(
            name='Color-Hidden Weights',
            matrix=CH_Weights_matrix,
        )
        WH_Weights = MappingProjection(
            name='Word-Hidden Weights',
            matrix=WH_Weights_matrix,
        )
        HO_Weights = MappingProjection(
            name='Hidden-Output Weights',
            matrix=HO_Weights_matrix,
        )

        color_naming_process = Process(
            default_variable=[1, 2.5],
            pathway=[colors, CH_Weights, hidden, HO_Weights, response],
            learning=LEARNING,
            target=[2, 2],
            name='Color Naming',
            prefs=process_prefs,
        )

        word_reading_process = Process(
            default_variable=[.5, 3],
            pathway=[words, WH_Weights, hidden],
            name='Word Reading',
            learning=LEARNING,
            target=[3, 3],
            prefs=process_prefs,
        )

        s = System(
            processes=[color_naming_process, word_reading_process],
            targets=[20, 20],
            name='Stroop Model',
            prefs=system_prefs,
        )

        def show_target():
            print('\nColor Naming\n\tInput: {}\n\tTarget: {}'.format([np.ndarray.tolist(item.parameters.value.get(s)) for item in colors.input_states], s.targets))
            print('Wording Reading:\n\tInput: {}\n\tTarget: {}\n'.format([np.ndarray.tolist(item.parameters.value.get(s)) for item in words.input_states], s.targets))
            print('Response: \n', response.output_state.parameters.value.get(s))
            print('Hidden-Output:')
            print(HO_Weights.get_mod_matrix(s))
            print('Color-Hidden:')
            print(CH_Weights.get_mod_matrix(s))
            print('Word-Hidden:')
            print(WH_Weights.get_mod_matrix(s))

        stim_list_dict = {
            colors: [[1, 1]],
            words: [[-2, -2]]
        }

        target_list_dict = {response: [[1, 1]]}

        results = s.run(
            num_trials=2,
            inputs=stim_list_dict,
            targets=target_list_dict,
            call_after_trial=show_target,
        )

        results_list = []
        for elem in s.results:
            for nested_elem in elem:
                nested_elem = nested_elem.tolist()
                try:
                    iter(nested_elem)
                except TypeError:
                    nested_elem = [nested_elem]
                results_list.extend(nested_elem)

        objective_response = s.mechanisms[3]
        objective_hidden = s.mechanisms[7]
        from pprint import pprint
        pprint(CH_Weights.__dict__)
        print(CH_Weights._parameter_states["matrix"].value)
        print(CH_Weights.get_mod_matrix(s))
        expected_output = [
            (colors.output_states[0].parameters.value.get(s), np.array([1., 1.])),
            (words.output_states[0].parameters.value.get(s), np.array([-2., -2.])),
            (hidden.output_states[0].parameters.value.get(s), np.array([0.13227553, 0.01990677])),
            (response.output_states[0].parameters.value.get(s), np.array([0.51044657, 0.5483048])),
            (objective_response.output_states[0].parameters.value.get(s), np.array([0.48955343, 0.4516952])),
            (objective_response.output_states[MSE].parameters.value.get(s), np.array(0.22184555903789838)),
            (CH_Weights.get_mod_matrix(s), np.array([
                [ 0.02512045, 1.02167245],
                [ 2.02512045, 3.02167245],
            ])),
            (WH_Weights.get_mod_matrix(s), np.array([
                [-0.05024091, 0.9566551 ],
                [ 1.94975909, 2.9566551 ],
            ])),
            (HO_Weights.get_mod_matrix(s), np.array([
                [ 0.03080958, 1.02830959],
                [ 2.00464242, 3.00426575],
            ])),
            (results, [[np.array([0.50899214, 0.54318254])], [np.array([0.51044657, 0.5483048])]]),
        ]

        for i in range(len(expected_output)):
            val, expected = expected_output[i]
            # setting absolute tolerance to be in accordance with reference_output precision
            # if you do not specify, assert_allcose will use a relative tolerance of 1e-07,
            # which WILL FAIL unless you gather higher precision values to use as reference
            np.testing.assert_allclose(val, expected, atol=1e-08, err_msg='Failed on expected_output[{0}]'.format(i))

        # KDM 10/16/18: Comparator Mechanism for Hidden is not executed by the system, because it's not associated with
        # an output mechanism. So it actually should be None instead of previously [0, 0] which was likely
        # a side effect with of conflation of different execution contexts
        assert objective_hidden.output_states[0].parameters.value.get(s) is None

    def test_bustamante_stroop_xor_lvoc(self):
        def w_fct(stim, color_control):
            '''function for word_task, to modulate strength of word reading based on 1-strength of color_naming ControlSignal'''
            return stim * (1 - color_control)
        w_fct_UDF = UserDefinedFunction(custom_function=w_fct, color_control=1)

        def objective_function(v):
            '''function used for ObjectiveMechanism of lvoc
             v[0] = output of DDM: [probability of color naming, probability of word reading]
             v[1] = reward:        [color naming rewarded, word reading rewarded]
             '''
            return np.sum(v[0] * v[1])

        color_stim = TransferMechanism(name='Color Stimulus', size=8)
        word_stim = TransferMechanism(name='Word Stimulus', size=8)

        color_task = TransferMechanism(name='Color Task')
        word_task = ProcessingMechanism(name='Word Task', function=w_fct_UDF)

        reward = TransferMechanism(name='Reward', size=2)

        task_decision = pnl.DDM(
            name='Task Decision',
            # function=NavarroAndFuss,
            output_states=[
                pnl.DDM_OUTPUT.PROBABILITY_UPPER_THRESHOLD,
                pnl.DDM_OUTPUT.PROBABILITY_LOWER_THRESHOLD
            ]
        )

        lvoc = pnl.LVOCControlMechanism(
            name='LVOC ControlMechanism',
            feature_predictors={pnl.SHADOW_EXTERNAL_INPUTS: [color_stim, word_stim]},
            objective_mechanism=pnl.ObjectiveMechanism(
                name='LVOC ObjectiveMechanism',
                monitored_output_states=[task_decision, reward],
                function=objective_function
            ),
            prediction_terms=[pnl.PV.FC, pnl.PV.COST],
            terminal_objective_mechanism=True,

            # learning_function=BayesGLM(mu_0=0, sigma_0=0.1),
            learning_function=BayesGLM,

            # function=GradientOptimization(
            #         convergence_criterion=VALUE,
            #         convergence_threshold=0.001,
            #         step_size=1,
            #         annealing_function= lambda x,y : x / np.sqrt(y),
            #         # direction=ASCENT
            # ),


            function=GridSearch,

            # function=OptimizationFunction,

            # control_signals={'COLOR CONTROL':[(SLOPE, color_task),
            #                                    ('color_control', word_task)]}
            # control_signals={NAME:'COLOR CONTROL',
            #                  PROJECTIONS:[(SLOPE, color_task),
            #                                   ('color_control', word_task)],
            #                  COST_OPTIONS:[ControlSignalCosts.INTENSITY,
            #                                    ControlSignalCosts.ADJUSTMENT],
            #                  INTENSITY_COST_FUNCTION:Exponential(rate=0.25, bias=-3),
            #                  ADJUSTMENT_COST_FUNCTION:Exponential(rate=0.25,bias=-3)}
            control_signals=pnl.ControlSignal(
                projections=[(pnl.SLOPE, color_task), ('color_control', word_task)],
                # function=ReLU,
                function=Logistic,
                cost_options=[pnl.ControlSignalCosts.INTENSITY, pnl.ControlSignalCosts.ADJUSTMENT],
                intensity_cost_function=Exponential(rate=0.25, bias=-3),
                adjustment_cost_function=Exponential(rate=0.25, bias=-3),
                allocation_samples=[i / 2 for i in list(range(0, 50, 1))]
            )
        )
        lvoc.reportOutputPref = True
        c = pnl.Composition(name='Stroop XOR Model')
        c.add_c_node(color_stim)
        c.add_c_node(word_stim)
        c.add_c_node(color_task, required_roles=pnl.CNodeRole.ORIGIN)
        c.add_c_node(word_task, required_roles=pnl.CNodeRole.ORIGIN)
        c.add_c_node(reward)
        c.add_c_node(task_decision)
        c.add_projection(sender=color_task, receiver=task_decision)
        c.add_projection(sender=word_task, receiver=task_decision)
        c.add_c_node(lvoc)

        # c.show_graph()

        input_dict = {color_stim: [[1, 0, 0, 0, 0, 0, 0, 0]],
                      word_stim: [[1, 0, 0, 0, 0, 0, 0, 0]],
                      color_task: [[1]],
                      word_task: [[-1]],
                      reward: [[1, 0]]}

        def run():
            c.run(inputs=input_dict, num_trials=1)

        run()
        run()

        control_signal_variables = [sig.parameters.variable.get(c) for sig in lvoc.control_signals]
        control_signal_values = [sig.parameters.value.get(c) for sig in lvoc.control_signals]
        features = lvoc.parameters.feature_values.get(c)
        lvoc_value = lvoc.compute_EVC([sig.parameters.variable.get(c) for sig in lvoc.control_signals], execution_id=c)

        print('\n')
        print('--------------------')
        print('ControlSignal variables: ', control_signal_variables)
        print('ControlSignal values: ', control_signal_values)
        print('features: ', features)
        print('lvoc: ', lvoc_value)
        print('--------------------')

        np.testing.assert_allclose([np.array([24.5])], control_signal_variables)
        np.testing.assert_allclose([np.array([1.])], control_signal_values)
        np.testing.assert_allclose(
            np.array([
                [1., 0., 0., 0., 0., 0., 0., 0.],
                [1., 0., 0., 0., 0., 0., 0., 0.]
            ]),
            features
        )
        np.testing.assert_allclose(np.array([0.14445302]), lvoc_value)
