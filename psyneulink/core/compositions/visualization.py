import inspect
import types
import typing
import warnings

import numpy as np
import typecheck as tc
from PIL import Image

from psyneulink._version import root_dir
from psyneulink.core.components.mechanisms.modulatory.control.controlmechanism import ControlMechanism
from psyneulink.core.components.mechanisms.modulatory.control.optimizationcontrolmechanism import AGENT_REP
from psyneulink.core.components.mechanisms.processing.compositioninterfacemechanism import (
    CompositionInterfaceMechanism
)
from psyneulink.core.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
from psyneulink.core.components.ports.inputport import InputPort
from psyneulink.core.components.ports.outputport import OutputPort
from psyneulink.core.components.projections.modulatory.controlprojection import ControlProjection
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.components.shellclasses import Component, Mechanism, Projection
from psyneulink.core.globals.context import ContextFlags, handle_external_context
from psyneulink.core.globals.keywords import (
    ALL, BOLD, BOTH, COMPONENT, CONDITIONS, EXECUTION_SET, FUNCTIONS, INITIAL_FRAME, LABELS,
    MECH_FUNCTION_PARAMS, MECHANISM, MECHANISMS, PORT_FUNCTION_PARAMS, PROJECTION, PROJECTIONS, ROLES, VALUES
)
from psyneulink.core.globals.utilities import convert_to_list, parse_string_to_psyneulink_object_string

__all__ = ['AnimationSettings', 'VisualizationError', 'show_graph']


class VisualizationError(Exception):
    pass


class AnimationSettings(types.SimpleNamespace):

    def __init__(
        self,
        unit: str = EXECUTION_SET,
        duration: typing.Union[int, float] = 0.75,
        num_runs: int = 1,
        num_trials: int = 1,
        simulations: bool = False,
        movie_name: typing.Optional[str] = None,
        movie_dir: typing.Optional[str] = None,
        save_images: bool = False,
        show: bool = False,
        show_cim: bool = False,
        show_controller: bool = False,
        composition_name: str = '',
        **addl_show_graph_args,
    ):
        if movie_name is None:
            movie_name = composition_name + '_movie' + '.gif'
        if movie_dir is None:
            movie_dir = root_dir + '/../show_graph output/GIFs/' + composition_name  # + " gifs"

        manual_enum_types = {'unit': {COMPONENT, EXECUTION_SET}}
        self._validate_constructor_args(
            {
                k: v for k, v in locals().items()
                if k not in {'self', 'manual_enum_types'}
            },
            manual_enum_types
        )

        super().__init__(
            unit=unit,
            duration=duration,
            num_runs=num_runs,
            num_trials=num_trials,
            simulations=simulations,
            movie_name=movie_name,
            movie_dir=movie_dir,
            save_images=save_images,
            show=show,
            show_cim=show_cim,
            show_controller=show_controller,
            composition_name=composition_name,
            addl_show_graph_args=addl_show_graph_args,
        )

    def _validate_constructor_args(self, arg_values, manual_enum_types):
        hints = typing.get_type_hints(self.__class__.__init__)
        message_template = "{0} of {1} entry of 'animate' argument ({2}) must be one of [{3}]"

        for name, value in arg_values.items():
            if name in manual_enum_types:
                is_valid = value in manual_enum_types[name]
                valid_types = []
                for item in manual_enum_types[name]:
                    keyword_val = parse_string_to_psyneulink_object_string(item)
                    if keyword_val is not None:
                        item = keyword_val
                    valid_types.append(str(item))

                valid_types = ', '.join(valid_types)
                category = 'value'
            elif name in hints:
                try:
                    is_valid = isinstance(value, hints[name])
                    valid_types = hints[name].__name__
                except TypeError:
                    # typing.Optional or Union
                    is_valid = isinstance(value, hints[name].__args__)
                    valid_types = ', '.join([str(x.__name__) for x in hints[name].__args__])
                category = 'type'
            else:
                continue

            if not is_valid:
                raise VisualizationError(
                    message_template.format(category, name, value, valid_types)
                )


@tc.typecheck
@handle_external_context(execution_id=NotImplemented, source=ContextFlags.COMPOSITION)
def show_graph(composition,
               show_node_structure:tc.any(bool, tc.enum(VALUES, LABELS, FUNCTIONS, MECH_FUNCTION_PARAMS,
                                                        PORT_FUNCTION_PARAMS, ROLES, ALL))=False,
               show_nested:tc.optional(tc.any(bool,dict,tc.enum(ALL)))=ALL,
               show_controller:tc.any(bool, tc.enum(AGENT_REP))=False,
               show_cim:bool=False,
               show_learning:bool=False,
               show_headers:bool=True,
               show_types:bool=False,
               show_dimensions:bool=False,
               show_projection_labels:bool=False,
               direction:tc.enum('BT', 'TB', 'LR', 'RL')='BT',
               # active_items:tc.optional(list)=None,
               active_items=None,
               active_color=BOLD,
               input_color='green',
               output_color='red',
               input_and_output_color='brown',
               # feedback_color='yellow',
               controller_color='blue',
               learning_color='orange',
               composition_color='pink',
               control_projection_arrow='box',
               feedback_shape = 'septagon',
               cim_shape='square',
               output_fmt:tc.optional(tc.enum('pdf','gv','jupyter','gif'))='pdf',
               context=None,
               **kwargs):
    """
    show_graph(                           \
       show_node_structure=False,         \
       show_nested=True,                  \
       show_controller=False,             \
       show_cim=False,                    \
       show_learning=False,               \
       show_headers=True,                 \
       show_types=False,                  \
       show_dimensions=False,             \
       show_projection_labels=False,      \
       direction='BT',                    \
       active_items=None,                 \
       active_color=BOLD,                 \
       input_color='green',               \
       output_color='red',                \
       input_and_output_color='brown',    \
       controller_color='blue',           \
       composition_color='pink',          \
       feedback_shape = 'septagon',       \
       cim_shape='square',                \
       output_fmt='pdf',                  \
       context=None)

    Show graphical display of Components in a Composition's graph.

    .. note::
       This method relies on `graphviz <http://www.graphviz.org>`_, which must be installed and imported
       (standard with PsyNeuLink pip install)

    See `Visualizing a Composition <Composition_Visualization>` for details and examples.

    Arguments
    ---------

    show_node_structure : bool, VALUES, LABELS, FUNCTIONS, MECH_FUNCTION_PARAMS, PORT_FUNCTION_PARAMS, ROLES, \
    or ALL : default False
        show a detailed representation of each `Mechanism <Mechanism>` in the graph, including its `Ports <Port>`;
        can have any of the following settings alone or in a list:

        * `True` -- show Ports of Mechanism, but not information about the `value
          <Component.value>` or `function <Component.function>` of the Mechanism or its Ports.

        * *VALUES* -- show the `value <Mechanism_Base.value>` of the Mechanism and the `value
          <Port_Base.value>` of each of its Ports.

        * *LABELS* -- show the `value <Mechanism_Base.value>` of the Mechanism and the `value
          <Port_Base.value>` of each of its Ports, using any labels for the values of InputPorts and
          OutputPorts specified in the Mechanism's `input_labels_dict <Mechanism.input_labels_dict>` and
          `output_labels_dict <Mechanism.output_labels_dict>`, respectively.

        * *FUNCTIONS* -- show the `function <Mechanism_Base.function>` of the Mechanism and the `function
          <Port_Base.function>` of its InputPorts and OutputPorts.

        * *MECH_FUNCTION_PARAMS_* -- show the parameters of the `function <Mechanism_Base.function>` for each
          Mechanism in the Composition (only applies if *FUNCTIONS* is True).

        * *PORT_FUNCTION_PARAMS_* -- show the parameters of the `function <Mechanism_Base.function>` for each
          Port of each Mechanism in the Composition (only applies if *FUNCTIONS* is True).

        * *ROLES* -- show the `role <NodeRole>` of the Mechanism in the Composition
          (but not any of the other information;  use *ALL* to show ROLES with other information).

        * *ALL* -- shows the role, `function <Component.function>`, and `value <Component.value>` of the
          Mechanisms in the `Composition` and their `Ports <Port>` (using labels for
          the values, if specified -- see above), including parameters for all functions.

    show_nested : bool | dict : default ALL
        specifies whether any nested Composition(s) are shown in details as inset graphs.  A dict can be used to
        specify any of the arguments allowed for show_graph to be used for the nested Composition(s);  *ALL*
        passes all arguments specified for the main Composition to the nested one(s);  True uses the default
        values of show_graph args for the nested Composition(s).

    show_controller :  bool or AGENT_REP : default False
        specifies whether or not to show the Composition's `controller <Composition.controller>` and associated
        `objective_mechanism <ControlMechanism.objective_mechanism>` if it has one.  If the controller is an
        OptimizationControlMechanism and it has an `agent_rep <OptimizationControlMechanism>`, then specifying
        *AGENT_REP* will also show that.  All of these items are displayed in the color specified for
        **controller_color**.

    show_cim : bool : default False
        specifies whether or not to show the Composition's input and out CompositionInterfaceMechanisms (CIMs)

    show_learning : bool or ALL : default False
        specifies whether or not to show the `learning components <Composition_Learning_Components>` of the
        `Composition`; they will all be displayed in the color specified for **learning_color**.
        Projections that receive a `LearningProjection` will be shown as a diamond-shaped node.
        If set to *ALL*, all Projections associated with learning will be shown:  the LearningProjections
        as well as from `ProcessingMechanisms <ProcessingMechanism>` to `LearningMechanisms <LearningMechanism>`
        that convey error and activation information;  if set to `True`, only the LearningPojections are shown.

    show_projection_labels : bool : default False
        specifies whether or not to show names of projections.

    show_headers : bool : default True
        specifies whether or not to show headers in the subfields of a Mechanism's node;  only takes effect if
        **show_node_structure** is specified (see above).

    show_types : bool : default False
        specifies whether or not to show type (class) of `Mechanism <Mechanism>` in each node label.

    show_dimensions : bool : default False
        specifies whether or not to show dimensions for the `variable <Component.variable>` and `value
        <Component.value>` of each Component in the graph (and/or MappingProjections when show_learning
        is `True`);  can have the following settings:

        * *MECHANISMS* -- shows `Mechanism <Mechanism>` input and output dimensions.  Input dimensions are shown
          in parentheses below the name of the Mechanism; each number represents the dimension of the `variable
          <InputPort.variable>` for each `InputPort` of the Mechanism; Output dimensions are shown above
          the name of the Mechanism; each number represents the dimension for `value <OutputPort.value>` of each
          of `OutputPort` of the Mechanism.

        * *PROJECTIONS* -- shows `MappingProjection` `matrix <MappingProjection.matrix>` dimensions.  Each is
          shown in (<dim>x<dim>...) format;  for standard 2x2 "weight" matrix, the first entry is the number of
          rows (input dimension) and the second the number of columns (output dimension).

        * *ALL* -- eqivalent to `True`; shows dimensions for both Mechanisms and Projections (see above for
          formats).

    direction : keyword : default 'BT'
        'BT': bottom to top; 'TB': top to bottom; 'LR': left to right; and 'RL`: right to left.

    active_items : List[Component] : default None
        specifies one or more items in the graph to display in the color specified by *active_color**.

    active_color : keyword : default 'yellow'
        specifies how to highlight the item(s) specified in *active_items**:  either a color recognized
        by GraphViz, or the keyword *BOLD*.

    input_color : keyword : default 'green',
        specifies the display color for `INPUT <NodeRole.INPUT>` Nodes in the Composition

    output_color : keyword : default 'red',
        specifies the display color for `OUTPUT` Nodes in the Composition

    input_and_output_color : keyword : default 'brown'
        specifies the display color of nodes that are both an `INPUT <NodeRole.INPUT>` and an `OUTPUT
        <NodeRole.OUTPUT>` Node in the Composition

    COMMENT:
    feedback_color : keyword : default 'yellow'
        specifies the display color of nodes that are assigned the `NodeRole` `FEEDBACK_SENDER`.
    COMMENT

    controller_color : keyword : default 'blue'
        specifies the color in which the controller components are displayed

    learning_color : keyword : default 'orange'
        specifies the color in which the learning components are displayed

    composition_color : keyword : default 'brown'
        specifies the display color of nodes that represent nested Compositions.

    feedback_shape : keyword : default 'septagon'
        specifies the display shape of nodes that are assigned the `NodeRole` `FEEDBACK_SENDER`.

    cim_shape : default 'square'
        specifies the display color input_CIM and output_CIM nodes

    output_fmt : keyword or None : default 'pdf'
        'pdf': generate and open a pdf with the visualization;
        'jupyter': return the object (for working in jupyter/ipython notebooks);
        'gv': return graphviz object
        'gif': return gif used for animation
        None : return None

    Returns
    -------

    `pdf` or Graphviz graph object :
        PDF: (placed in current directory) if :keyword:`output_fmt` arg is 'pdf';
        Graphviz graph object if :keyword:`output_fmt` arg is 'gv' or 'jupyter';
        gif if :keyword:`output_fmt` arg is 'gif'.

    """

    # HELPER METHODS ----------------------------------------------------------------------

    tc.typecheck
    _locals = locals().copy()

    from psyneulink.core.compositions.composition import Composition, NodeRole

    def _assign_processing_components(g, rcvr, show_nested):
        """Assign nodes to graph"""
        if isinstance(rcvr, Composition) and show_nested:
            # User passed args for nested Composition
            output_fmt_arg = {'output_fmt':'gv'}
            if isinstance(show_nested, dict):
                args = show_nested
                args.update(output_fmt_arg)
            elif show_nested == ALL:
                # Pass args from main call to show_graph to call for nested Composition
                args = dict({k:_locals[k] for k in list(inspect.signature(show_graph).parameters)})
                args.update(output_fmt_arg)
                if kwargs:
                    args['kwargs'] = kwargs
                else:
                    del  args['kwargs']
            else:
                # Use default args for nested Composition
                args = output_fmt_arg
            nested_comp_graph = rcvr.show_graph(**args)
            nested_comp_graph.name = "cluster_" + rcvr.name
            rcvr_label = rcvr.name
            # if rcvr in composition.get_nodes_by_role(NodeRole.FEEDBACK_SENDER):
            #     nested_comp_graph.attr(color=feedback_color)
            if rcvr in composition.get_nodes_by_role(NodeRole.INPUT) and \
                    rcvr in composition.get_nodes_by_role(NodeRole.OUTPUT):
                nested_comp_graph.attr(color=input_and_output_color)
            elif rcvr in composition.get_nodes_by_role(NodeRole.INPUT):
                nested_comp_graph.attr(color=input_color)
            elif rcvr in composition.get_nodes_by_role(NodeRole.OUTPUT):
                nested_comp_graph.attr(color=output_color)
            nested_comp_graph.attr(label=rcvr_label)
            g.subgraph(nested_comp_graph)

        # If rcvr is a learning component and not an INPUT node,
        #    break and handle in _assign_learning_components()
        #    (node: this allows TARGET node for learning to remain marked as an INPUT node)
        if (NodeRole.LEARNING in composition.nodes_to_roles[rcvr]
                and not NodeRole.INPUT in composition.nodes_to_roles[rcvr]):
            return

        # If rcvr is ObjectiveMechanism for Composition's controller,
        #    break and handle in _assign_control_components()
        if (isinstance(rcvr, ObjectiveMechanism)
                and composition.controller
                and rcvr is composition.controller.objective_mechanism):
            return

        # Implement rcvr node
        else:

            # Set rcvr shape, color, and penwidth based on node type
            rcvr_rank = 'same'

            # Feedback Node
            if rcvr in composition.get_nodes_by_role(NodeRole.FEEDBACK_SENDER):
                node_shape = feedback_shape
            else:
                node_shape = mechanism_shape

            # Get condition if any associated with rcvr
            if rcvr in composition.scheduler.conditions:
                condition = composition.scheduler.conditions[rcvr]
            else:
                condition = None

            # # Feedback Node
            # if rcvr in composition.get_nodes_by_role(NodeRole.FEEDBACK_SENDER):
            #     if rcvr in active_items:
            #         if active_color == BOLD:
            #             rcvr_color = feedback_color
            #         else:
            #             rcvr_color = active_color
            #         rcvr_penwidth = str(bold_width + active_thicker_by)
            #         composition.active_item_rendered = True
            #     else:
            #         rcvr_color = feedback_color
            #         rcvr_penwidth = str(bold_width)

            # Input and Output Node
            if rcvr in composition.get_nodes_by_role(NodeRole.INPUT) and \
                    rcvr in composition.get_nodes_by_role(NodeRole.OUTPUT):
                if rcvr in active_items:
                    if active_color == BOLD:
                        rcvr_color = input_and_output_color
                    else:
                        rcvr_color = active_color
                    rcvr_penwidth = str(bold_width + active_thicker_by)
                    composition.active_item_rendered = True
                else:
                    rcvr_color = input_and_output_color
                    rcvr_penwidth = str(bold_width)

            # Input Node
            elif rcvr in composition.get_nodes_by_role(NodeRole.INPUT):
                if rcvr in active_items:
                    if active_color == BOLD:
                        rcvr_color = input_color
                    else:
                        rcvr_color = active_color
                    rcvr_penwidth = str(bold_width + active_thicker_by)
                    composition.active_item_rendered = True
                else:
                    rcvr_color = input_color
                    rcvr_penwidth = str(bold_width)
                rcvr_rank = input_rank

            # Output Node
            elif rcvr in composition.get_nodes_by_role(NodeRole.OUTPUT):
                if rcvr in active_items:
                    if active_color == BOLD:
                        rcvr_color = output_color
                    else:
                        rcvr_color = active_color
                    rcvr_penwidth = str(bold_width + active_thicker_by)
                    composition.active_item_rendered = True
                else:
                    rcvr_color = output_color
                    rcvr_penwidth = str(bold_width)
                rcvr_rank = output_rank

            # Composition
            elif isinstance(rcvr, Composition):
                node_shape = composition_shape
                if rcvr in active_items:
                    if active_color == BOLD:
                        rcvr_color = composition_color
                    else:
                        rcvr_color = active_color
                    rcvr_penwidth = str(bold_width + active_thicker_by)
                    composition.active_item_rendered = True
                else:
                    rcvr_color = composition_color
                    rcvr_penwidth = str(bold_width)

            elif rcvr in active_items:
                if active_color == BOLD:
                    rcvr_color = default_node_color
                else:
                    rcvr_color = active_color
                rcvr_penwidth = str(default_width + active_thicker_by)
                composition.active_item_rendered = True

            else:
                rcvr_color = default_node_color
                rcvr_penwidth = str(default_width)

            # Implement rcvr node
            rcvr_label = _get_graph_node_label(rcvr,
                                                    show_types,
                                                    show_dimensions)

            if show_node_structure and isinstance(rcvr, Mechanism):
                g.node(rcvr_label,
                       rcvr._show_structure(**node_struct_args, node_border=rcvr_penwidth, condition=condition),
                       shape=struct_shape,
                       color=rcvr_color,
                       rank=rcvr_rank,
                       penwidth=rcvr_penwidth)
            else:
                g.node(rcvr_label,
                       shape=node_shape,
                       color=rcvr_color,
                       rank=rcvr_rank,
                       penwidth=rcvr_penwidth)

        # Implement sender edges
        sndrs = processing_graph[rcvr]
        _assign_incoming_edges(g, rcvr, rcvr_label, sndrs)

    def _assign_cim_components(g, cims):

        cim_rank = 'same'

        for cim in cims:

            cim_penwidth = str(default_width)

            # ASSIGN CIM NODE ****************************************************************

            # Assign color
            # Also take opportunity to verify that cim is either input_CIM or output_CIM
            if cim is composition.input_CIM:
                if cim in active_items:
                    if active_color == BOLD:
                        cim_color = input_color
                    else:
                        cim_color = active_color
                    cim_penwidth = str(default_width + active_thicker_by)
                    composition.active_item_rendered = True
                else:
                    cim_color = input_color

            elif cim is composition.output_CIM:
                if cim in active_items:
                    if active_color == BOLD:
                        cim_color = output_color
                    else:
                        cim_color = active_color
                    cim_penwidth = str(default_width + active_thicker_by)
                    composition.active_item_rendered = True
                else:
                    cim_color = output_color

            else:
                assert False, '_assignm_cim_components called with node that is not input_CIM or output_CIM'

            # Assign lablel
            cim_label = _get_graph_node_label(cim, show_types, show_dimensions)

            if show_node_structure:
                g.node(cim_label,
                       cim._show_structure(**node_struct_args, node_border=cim_penwidth, compact_cim=True),
                       shape=struct_shape,
                       color=cim_color,
                       rank=cim_rank,
                       penwidth=cim_penwidth)

            else:
                g.node(cim_label,
                       shape=cim_shape,
                       color=cim_color,
                       rank=cim_rank,
                       penwidth=cim_penwidth)

            # ASSIGN CIM PROJECTIONS ****************************************************************

            # Projections from input_CIM to INPUT nodes
            if cim is composition.input_CIM:

                for output_port in composition.input_CIM.output_ports:
                    projs = output_port.efferents
                    for proj in projs:
                        input_mech = proj.receiver.owner
                        if input_mech is composition.controller:
                            # Projections to contoller are handled under _assign_controller_components
                            continue
                        # Validate the Projection is to an INPUT node or a node that is shadowing one
                        if ((input_mech in composition.nodes_to_roles and
                             not NodeRole.INPUT in composition.nodes_to_roles[input_mech])
                                and (proj.receiver.shadow_inputs in composition.nodes_to_roles and
                                     not NodeRole.INPUT in composition.nodes_to_roles[proj.receiver.shadow_inputs])):
                            raise VisualizationError("Projection from input_CIM of {} to node {} "
                                                   "that is not an {} node or shadowing its {}".
                                                   format(composition.name, input_mech,
                                                          NodeRole.INPUT.name, NodeRole.INPUT.name.lower()))
                        # Construct edge name
                        input_mech_label = _get_graph_node_label(input_mech,
                                                                      show_types,
                                                                      show_dimensions)
                        if show_node_structure:
                            cim_proj_label = '{}:{}-{}'. \
                                format(cim_label, OutputPort.__name__, proj.sender.name)
                            proc_mech_rcvr_label = '{}:{}-{}'. \
                                format(input_mech_label, InputPort.__name__, proj.receiver.name)
                        else:
                            cim_proj_label = cim_label
                            proc_mech_rcvr_label = input_mech_label

                        # Render Projection
                        if any(item in active_items for item in {proj, proj.receiver.owner}):
                            if active_color == BOLD:
                                proj_color = default_node_color
                            else:
                                proj_color = active_color
                            proj_width = str(default_width + active_thicker_by)
                            composition.active_item_rendered = True
                        else:
                            proj_color = default_node_color
                            proj_width = str(default_width)
                        if show_projection_labels:
                            label = _get_graph_node_label(proj, show_types, show_dimensions)
                        else:
                            label = ''
                        g.edge(cim_proj_label, proc_mech_rcvr_label, label=label,
                               color=proj_color, penwidth=proj_width)

            # Projections from OUTPUT nodes to output_CIM
            if cim is composition.output_CIM:
                # Construct edge name
                for input_port in composition.output_CIM.input_ports:
                    projs = input_port.path_afferents
                    for proj in projs:
                        # Validate the Projection is from an OUTPUT node
                        output_mech = proj.sender.owner
                        if not NodeRole.OUTPUT in composition.nodes_to_roles[output_mech]:
                            raise VisualizationError("Projection to output_CIM of {} from node {} "
                                                   "that is not an {} node".
                                                   format(composition.name, output_mech,
                                                          NodeRole.OUTPUT.name, NodeRole.OUTPUT.name.lower()))
                        # Construct edge name
                        output_mech_label = _get_graph_node_label(output_mech,
                                                                       show_types,
                                                                       show_dimensions)
                        if show_node_structure:
                            cim_proj_label = '{}:{}'. \
                                format(cim_label, cim._get_port_name(proj.receiver))
                            proc_mech_sndr_label = '{}:{}'.\
                                format(output_mech_label, output_mech._get_port_name(proj.sender))
                                # format(output_mech_label, OutputPort.__name__, proj.sender.name)
                        else:
                            cim_proj_label = cim_label
                            proc_mech_sndr_label = output_mech_label

                        # Render Projection
                        if any(item in active_items for item in {proj, proj.receiver.owner}):
                            if active_color == BOLD:
                                proj_color = default_node_color
                            else:
                                proj_color = active_color
                            proj_width = str(default_width + active_thicker_by)
                            composition.active_item_rendered = True
                        else:
                            proj_color = default_node_color
                            proj_width = str(default_width)
                        if show_projection_labels:
                            label = _get_graph_node_label(proj, show_types, show_dimensions)
                        else:
                            label = ''
                        g.edge(proc_mech_sndr_label, cim_proj_label, label=label,
                               color=proj_color, penwidth=proj_width)

    def _assign_controller_components(g):
        """Assign control nodes and edges to graph"""

        controller = composition.controller
        if controller is None:
            warnings.warn(f"{composition.name} has not been assigned a \'controller\', "
                          f"so \'show_controller\' option in call to its show_graph() method will be ignored.")
            return

        if controller in active_items:
            if active_color == BOLD:
                ctlr_color = controller_color
            else:
                ctlr_color = active_color
            ctlr_width = str(default_width + active_thicker_by)
            composition.active_item_rendered = True
        else:
            ctlr_color = controller_color
            ctlr_width = str(default_width)

        # Assign controller node
        node_shape = mechanism_shape
        ctlr_label = _get_graph_node_label(controller, show_types, show_dimensions)
        if show_node_structure:
            g.node(ctlr_label,
                   controller._show_structure(**node_struct_args, node_border=ctlr_width,
                                             condition=composition.controller_condition),
                   shape=struct_shape,
                   color=ctlr_color,
                   penwidth=ctlr_width,
                   rank=control_rank
                   )
        else:
            g.node(ctlr_label,
                    color=ctlr_color, penwidth=ctlr_width, shape=node_shape,
                    rank=control_rank)

        # outgoing edges (from controller to ProcessingMechanisms)
        for control_signal in controller.control_signals:
            for ctl_proj in control_signal.efferents:
                proc_mech_label = _get_graph_node_label(ctl_proj.receiver.owner, show_types, show_dimensions)
                if controller in active_items:
                    if active_color == BOLD:
                        ctl_proj_color = controller_color
                    else:
                        ctl_proj_color = active_color
                    ctl_proj_width = str(default_width + active_thicker_by)
                    composition.active_item_rendered = True
                else:
                    ctl_proj_color = controller_color
                    ctl_proj_width = str(default_width)
                if show_projection_labels:
                    edge_label = ctl_proj.name
                else:
                    edge_label = ''
                if show_node_structure:
                    ctl_sndr_label = ctlr_label + ':' + controller._get_port_name(control_signal)
                    proc_mech_rcvr_label = \
                        proc_mech_label + ':' + controller._get_port_name(ctl_proj.receiver)
                else:
                    ctl_sndr_label = ctlr_label
                    proc_mech_rcvr_label = proc_mech_label
                g.edge(ctl_sndr_label,
                       proc_mech_rcvr_label,
                       label=edge_label,
                       color=ctl_proj_color,
                       penwidth=ctl_proj_width
                       )

        # If controller has objective_mechanism, assign its node and Projections
        if controller.objective_mechanism:
            # get projection from ObjectiveMechanism to ControlMechanism
            objmech_ctlr_proj = controller.input_port.path_afferents[0]
            if controller in active_items:
                if active_color == BOLD:
                    objmech_ctlr_proj_color = controller_color
                else:
                    objmech_ctlr_proj_color = active_color
                objmech_ctlr_proj_width = str(default_width + active_thicker_by)
                composition.active_item_rendered = True
            else:
                objmech_ctlr_proj_color = controller_color
                objmech_ctlr_proj_width = str(default_width)

            # get ObjectiveMechanism
            objmech = objmech_ctlr_proj.sender.owner
            if objmech in active_items:
                if active_color == BOLD:
                    objmech_color = controller_color
                else:
                    objmech_color = active_color
                objmech_width = str(default_width + active_thicker_by)
                composition.active_item_rendered = True
            else:
                objmech_color = controller_color
                objmech_width = str(default_width)

            objmech_label = _get_graph_node_label(objmech, show_types, show_dimensions)
            if show_node_structure:
                if objmech in composition.scheduler.conditions:
                    condition = composition.scheduler.conditions[objmech]
                else:
                    condition = None
                g.node(objmech_label,
                       objmech._show_structure(**node_struct_args, node_border=ctlr_width, condition=condition),
                       shape=struct_shape,
                       color=objmech_color,
                       penwidth=ctlr_width,
                       rank=control_rank
                       )
            else:
                g.node(objmech_label,
                        color=objmech_color, penwidth=objmech_width, shape=node_shape,
                        rank=control_rank)

            # objmech to controller edge
            if show_projection_labels:
                edge_label = objmech_ctlr_proj.name
            else:
                edge_label = ''
            if show_node_structure:
                obj_to_ctrl_label = objmech_label + ':' + objmech._get_port_name(objmech_ctlr_proj.sender)
                ctlr_from_obj_label = ctlr_label + ':' + objmech._get_port_name(objmech_ctlr_proj.receiver)
            else:
                obj_to_ctrl_label = objmech_label
                ctlr_from_obj_label = ctlr_label
            g.edge(obj_to_ctrl_label, ctlr_from_obj_label, label=edge_label,
                   color=objmech_ctlr_proj_color, penwidth=objmech_ctlr_proj_width)

            # incoming edges (from monitored mechs to objective mechanism)
            for input_port in objmech.input_ports:
                for projection in input_port.path_afferents:
                    if objmech in active_items:
                        if active_color == BOLD:
                            proj_color = controller_color
                        else:
                            proj_color = active_color
                        proj_width = str(default_width + active_thicker_by)
                        composition.active_item_rendered = True
                    else:
                        proj_color = controller_color
                        proj_width = str(default_width)
                    if show_node_structure:
                        sndr_proj_label = _get_graph_node_label(projection.sender.owner,
                                                                     show_types,
                                                                     show_dimensions) + \
                                          ':' + objmech._get_port_name(projection.sender)
                        objmech_proj_label = objmech_label + ':' + objmech._get_port_name(input_port)
                    else:
                        sndr_proj_label = _get_graph_node_label(projection.sender.owner,
                                                                     show_types,
                                                                     show_dimensions)
                        objmech_proj_label = _get_graph_node_label(objmech,
                                                                        show_types,
                                                                        show_dimensions)
                    if show_projection_labels:
                        edge_label = projection.name
                    else:
                        edge_label = ''
                    g.edge(sndr_proj_label, objmech_proj_label, label=edge_label,
                           color=proj_color, penwidth=proj_width)

        # If controller has an agent_rep, assign its node and edges (not Projections per se)
        if hasattr(controller, 'agent_rep') and controller.agent_rep and show_controller==AGENT_REP :
            # get agent_rep
            agent_rep = controller.agent_rep
            # controller is active, treat
            if controller in active_items:
                if active_color == BOLD:
                    agent_rep_color = controller_color
                else:
                    agent_rep_color = active_color
                agent_rep_width = str(default_width + active_thicker_by)
                composition.active_item_rendered = True
            else:
                agent_rep_color = controller_color
                agent_rep_width = str(default_width)

            # agent_rep node
            agent_rep_label = _get_graph_node_label(agent_rep, show_types, show_dimensions)
            g.node(agent_rep_label,
                    color=agent_rep_color, penwidth=agent_rep_width, shape=agent_rep_shape,
                    rank=control_rank)

            # agent_rep <-> controller edges
            g.edge(agent_rep_label, ctlr_label, color=agent_rep_color, penwidth=agent_rep_width)
            g.edge(ctlr_label, agent_rep_label, color=agent_rep_color, penwidth=agent_rep_width)

        # get any other incoming edges to controller (i.e., other than from ObjectiveMechanism)
        senders = set()
        for i in controller.input_ports[1:]:
            for p in i.path_afferents:
                senders.add(p.sender.owner)
        _assign_incoming_edges(g, controller, ctlr_label, senders, proj_color=ctl_proj_color)

    def _assign_learning_components(g):
        """Assign learning nodes and edges to graph"""

        # Get learning_components, with exception of INPUT (i.e. TARGET) nodes
        #    (i.e., allow TARGET node to continue to be marked as an INPUT node)
        learning_components = [node for node in composition.learning_components
                               if not NodeRole.INPUT in composition.nodes_to_roles[node]]
        # learning_components.extend([node for node in composition.nodes if
        #                             NodeRole.AUTOASSOCIATIVE_LEARNING in
        #                             composition.nodes_to_roles[node]])

        for rcvr in learning_components:
            # if rcvr is Projection, skip (handled in _assign_processing_components)
            if isinstance(rcvr, MappingProjection):
                return

            # Get rcvr info
            rcvr_label = _get_graph_node_label(rcvr, show_types, show_dimensions)
            if rcvr in active_items:
                if active_color == BOLD:
                    rcvr_color = learning_color
                else:
                    rcvr_color = active_color
                rcvr_width = str(default_width + active_thicker_by)
                composition.active_item_rendered = True
            else:
                rcvr_color = learning_color
                rcvr_width = str(default_width)

            # rcvr is a LearningMechanism or ObjectiveMechanism (ComparatorMechanism)
            # Implement node for Mechanism
            if show_node_structure:
                g.node(rcvr_label,
                        rcvr._show_structure(**node_struct_args),
                        rank=learning_rank, color=rcvr_color, penwidth=rcvr_width)
            else:
                g.node(rcvr_label,
                        color=rcvr_color, penwidth=rcvr_width,
                        rank=learning_rank, shape=mechanism_shape)

            # Implement sender edges
            sndrs = processing_graph[rcvr]
            _assign_incoming_edges(g, rcvr, rcvr_label, sndrs)

    def render_projection_as_node(g, proj, label,
                                  proj_color, proj_width,
                                  sndr_label=None,
                                  rcvr_label=None):

        proj_receiver = proj.receiver.owner

        # Node for Projection
        g.node(label, shape=learning_projection_shape, color=proj_color, penwidth=proj_width)

        # FIX: ??
        if proj_receiver in active_items:
            # edge_color = proj_color
            # edge_width = str(proj_width)
            if active_color == BOLD:
                edge_color = proj_color
            else:
                edge_color = active_color
            edge_width = str(default_width + active_thicker_by)
        else:
            edge_color = default_node_color
            edge_width = str(default_width)

        # Edges to and from Projection node
        if sndr_label:
            G.edge(sndr_label, label, arrowhead='none',
                   color=edge_color, penwidth=edge_width)
        if rcvr_label:
            G.edge(label, rcvr_label,
                   color=edge_color, penwidth=edge_width)

        # LearningProjection(s) to node
        # if proj in active_items or (proj_learning_in_execution_phase and proj_receiver in active_items):
        if proj in active_items:
            if active_color == BOLD:
                learning_proj_color = learning_color
            else:
                learning_proj_color = active_color
            learning_proj_width = str(default_width + active_thicker_by)
            composition.active_item_rendered = True
        else:
            learning_proj_color = learning_color
            learning_proj_width = str(default_width)
        sndrs = proj._parameter_ports['matrix'].mod_afferents # GET ALL LearningProjections to proj
        for sndr in sndrs:
            sndr_label = _get_graph_node_label(sndr.sender.owner, show_types, show_dimensions)
            rcvr_label = _get_graph_node_label(proj, show_types, show_dimensions)
            if show_projection_labels:
                edge_label = proj._parameter_ports['matrix'].mod_afferents[0].name
            else:
                edge_label = ''
            if show_node_structure:
                G.edge(sndr_label + ':' + OutputPort.__name__ + '-' + 'LearningSignal',
                       rcvr_label,
                       label=edge_label,
                       color=learning_proj_color, penwidth=learning_proj_width)
            else:
                G.edge(sndr_label, rcvr_label, label = edge_label,
                       color=learning_proj_color, penwidth=learning_proj_width)
        return True

    @tc.typecheck
    def _assign_incoming_edges(g, rcvr, rcvr_label, senders, proj_color=None, proj_arrow=None):

        proj_color = proj_color or default_node_color
        proj_arrow = default_projection_arrow

        for sndr in senders:

            # Set sndr info
            sndr_label = _get_graph_node_label(sndr, show_types, show_dimensions)

            # Iterate through all Projections from all OutputPorts of sndr
            for output_port in sndr.output_ports:
                for proj in output_port.efferents:

                    # Skip any projections to ObjectiveMechanism for controller
                    #   (those are handled in _assign_control_components)
                    if (composition.controller and
                            proj.receiver.owner in {composition.controller, composition.controller.objective_mechanism}):
                        continue

                    # Only consider Projections to the rcvr
                    if ((isinstance(rcvr, (Mechanism, Projection)) and proj.receiver.owner == rcvr)
                            or (isinstance(rcvr, Composition) and proj.receiver.owner is rcvr.input_CIM)):

                        if show_node_structure and isinstance(sndr, Mechanism) and isinstance(rcvr, Mechanism):
                            sndr_proj_label = f'{sndr_label}:{sndr._get_port_name(proj.sender)}'
                            proc_mech_rcvr_label = f'{rcvr_label}:{rcvr._get_port_name(proj.receiver)}'
                        else:
                            sndr_proj_label = sndr_label
                            proc_mech_rcvr_label = rcvr_label
                        try:
                            has_learning = proj.has_learning_projection is not None
                        except AttributeError:
                            has_learning = None

                        edge_label = _get_graph_node_label(proj, show_types, show_dimensions)
                        is_learning_component = rcvr in composition.learning_components or sndr in composition.learning_components

                        # Check if Projection or its receiver is active
                        if any(item in active_items for item in {proj, proj.receiver.owner}):
                            if active_color == BOLD:
                                # if (isinstance(rcvr, LearningMechanism) or isinstance(sndr, LearningMechanism)):
                                if is_learning_component:
                                    proj_color = learning_color
                                else:
                                    pass
                            else:
                                proj_color = active_color
                            proj_width = str(default_width + active_thicker_by)
                            composition.active_item_rendered = True

                        # Projection to or from a LearningMechanism
                        elif (NodeRole.LEARNING in composition.nodes_to_roles[rcvr]):
                            proj_color = learning_color
                            proj_width = str(default_width)

                        else:
                            proj_width = str(default_width)
                        proc_mech_label = edge_label

                        # Render Projection as edge

                        if show_learning and has_learning:
                            # Render Projection as node
                            #    (do it here rather than in _assign_learning_components,
                            #     as it needs afferent and efferent edges to other nodes)
                            # IMPLEMENTATION NOTE: Projections can't yet use structured nodes:
                            deferred = not render_projection_as_node(g=g, proj=proj,
                                                                     label=proc_mech_label,
                                                                     rcvr_label=proc_mech_rcvr_label,
                                                                     sndr_label=sndr_proj_label,
                                                                     proj_color=proj_color,
                                                                     proj_width=proj_width)
                            # Deferred if it is the last Mechanism in a learning Pathway
                            # (see _render_projection_as_node)
                            if deferred:
                                continue

                        else:
                            from psyneulink.core.components.projections.modulatory.controlprojection \
                                import ControlProjection
                            if isinstance(proj, ControlProjection):
                                arrowhead=control_projection_arrow
                            else:
                                arrowhead=proj_arrow
                            if show_projection_labels:
                                label = proc_mech_label
                            else:
                                label = ''
                            g.edge(sndr_proj_label, proc_mech_rcvr_label,
                                   label=label,
                                   color=proj_color,
                                   penwidth=proj_width,
                                   arrowhead=arrowhead)

    # SETUP AND CONSTANTS -----------------------------------------------------------------

    if context.execution_id is NotImplemented:
        context.execution_id = composition.default_execution_id

    # For backward compatibility
    if 'show_model_based_optimizer' in kwargs:
        show_controller = kwargs['show_model_based_optimizer']
        del kwargs['show_model_based_optimizer']
    if kwargs:
        raise VisualizationError(f'Unrecognized argument(s) in call to show_graph method '
                               f'of {Composition.__name__} {repr(composition.name)}: {", ".join(kwargs.keys())}')

    if show_dimensions == True:
        show_dimensions = ALL

    active_items = active_items or []
    if active_items:
        active_items = convert_to_list(active_items)
        if (
            composition.scheduler.get_clock(context).time.run >= composition._animate.num_runs
            or composition.scheduler.get_clock(context).time.trial >= composition._animate.num_trials
        ):
            return

        for item in active_items:
            if not isinstance(item, Component) and item is not INITIAL_FRAME:
                raise VisualizationError(
                    "PROGRAM ERROR: Item ({}) specified in {} argument for {} method of {} is not a {}".
                    format(item, repr('active_items'), repr('show_graph'), composition.name, Component.__name__))

    composition.active_item_rendered = False

    # Argument values used to call Mechanism._show_structure()
    if isinstance(show_node_structure, (list, tuple, set)):
        node_struct_args = {'composition': composition,
                            'show_roles': any(key in show_node_structure for key in {ROLES, ALL}),
                            'show_conditions': any(key in show_node_structure for key in {CONDITIONS, ALL}),
                            'show_functions': any(key in show_node_structure for key in {FUNCTIONS, ALL}),
                            'show_mech_function_params': any(key in show_node_structure
                                                             for key in {MECH_FUNCTION_PARAMS, ALL}),
                            'show_port_function_params': any(key in show_node_structure
                                                              for key in {PORT_FUNCTION_PARAMS, ALL}),
                            'show_values': any(key in show_node_structure for key in {VALUES, ALL}),
                            'use_labels': any(key in show_node_structure for key in {LABELS, ALL}),
                            'show_headers': show_headers,
                            'output_fmt': 'struct',
                            'context':context}
    else:
        node_struct_args = {'composition': composition,
                            'show_roles': show_node_structure in {ROLES, ALL},
                            'show_conditions': show_node_structure in {CONDITIONS, ALL},
                            'show_functions': show_node_structure in {FUNCTIONS, ALL},
                            'show_mech_function_params': show_node_structure in {MECH_FUNCTION_PARAMS, ALL},
                            'show_port_function_params': show_node_structure in {PORT_FUNCTION_PARAMS, ALL},
                            'show_values': show_node_structure in {VALUES, LABELS, ALL},
                            'use_labels': show_node_structure in {LABELS, ALL},
                            'show_headers': show_headers,
                            'output_fmt': 'struct',
                            'context': context}

    # DEFAULT ATTRIBUTES ----------------------------------------------------------------

    default_node_color = 'black'
    mechanism_shape = 'oval'
    learning_projection_shape = 'diamond'
    struct_shape = 'plaintext' # assumes use of html
    cim_shape = 'rectangle'
    composition_shape = 'rectangle'
    agent_rep_shape = 'egg'
    default_projection_arrow = 'normal'

    bold_width = 3
    default_width = 1
    active_thicker_by = 2

    input_rank = 'source'
    control_rank = 'min'
    learning_rank = 'min'
    output_rank = 'max'

    # BUILD GRAPH ------------------------------------------------------------------------

    import graphviz as gv

    G = gv.Digraph(
        name=composition.name,
        engine="dot",
        node_attr={
            'fontsize': '12',
            'fontname': 'arial',
            'shape': 'record',
            'color': default_node_color,
            'penwidth': str(default_width),
        },
        edge_attr={
            'fontsize': '10',
            'fontname': 'arial'
        },
        graph_attr={
            "rankdir": direction,
            'overlap': "False"
        },
    )

    # get all Nodes
    # FIX: call to _analyze_graph in nested calls to show_graph cause trouble
    if output_fmt != 'gv':
        composition._analyze_graph(context=context)
    processing_graph = composition.graph_processing.dependency_dict
    rcvrs = list(processing_graph.keys())

    for r in rcvrs:
        _assign_processing_components(G, r, show_nested)

    # Add cim Components to graph if show_cim
    if show_cim:
        _assign_cim_components(G, [composition.input_CIM, composition.output_CIM])

    # Add controller-related Components to graph if show_controller
    if show_controller:
        _assign_controller_components(G)

    # Add learning-related Components to graph if show_learning
    if show_learning:
        _assign_learning_components(G)

    # Sort nodes for display
    def get_index_of_node_in_G_body(node, node_type:tc.enum(MECHANISM, PROJECTION, BOTH)):
        """Get index of node in G.body"""
        for i, item in enumerate(G.body):
            if node.name in item:
                if node_type in {MECHANISM, BOTH}:
                    if not '->' in item:
                        return i
                elif node_type in {PROJECTION, BOTH}:
                    if '->' in item:
                        return i
                else:
                    assert False, f'PROGRAM ERROR: node_type not specified or illegal ({node_type})'

    for node in composition.nodes:
        if isinstance(node, Composition):
            continue
        roles = composition.get_roles_by_node(node)
        # Put INPUT node(s) first
        if NodeRole.INPUT in roles:
            i = get_index_of_node_in_G_body(node, MECHANISM)
            if i is not None:
                G.body.insert(0,G.body.pop(i))
        # Put OUTPUT node(s) last (except for ControlMechanisms)
        if NodeRole.OUTPUT in roles:
            i = get_index_of_node_in_G_body(node, MECHANISM)
            if i is not None:
                G.body.insert(len(G.body),G.body.pop(i))
        # Put ControlMechanism(s) last
        if isinstance(node, ControlMechanism):
            i = get_index_of_node_in_G_body(node, MECHANISM)
            if i is not None:
                G.body.insert(len(G.body),G.body.pop(i))

    for proj in composition.projections:
        # Put ControlProjection(s) last (along with ControlMechanis(s))
        if isinstance(proj, ControlProjection):
            i = get_index_of_node_in_G_body(node, PROJECTION)
            if i is not None:
                G.body.insert(len(G.body),G.body.pop(i))

    if composition.controller and show_controller:
        i = get_index_of_node_in_G_body(composition.controller, MECHANISM)
        G.body.insert(len(G.body),G.body.pop(i))

    # GENERATE OUTPUT ---------------------------------------------------------------------

    # Show as pdf
    try:
        if output_fmt == 'pdf':
            # G.format = 'svg'
            G.view(composition.name.replace(" ", "-"), cleanup=True, directory='show_graph OUTPUT/PDFS')

        # Generate images for animation
        elif output_fmt == 'gif':
            if composition.active_item_rendered or INITIAL_FRAME in active_items:
                _generate_gifs(composition, G, active_items, context)

        # Return graph to show in jupyter
        elif output_fmt == 'jupyter':
            return G

        elif output_fmt == 'gv':
            return G
        elif not output_fmt:
            return None
        else:
            raise VisualizationError(f"Bad arg in call to {self.name}.show_graph: '{output_fmt}'.")

    except:
        raise VisualizationError(f"Problem displaying graph for {composition.name}")


def _get_graph_node_label(item, show_types=None, show_dimensions=None):
    from psyneulink.core.compositions.composition import Composition

    if not isinstance(item, (Mechanism, Composition, Projection)):
        raise VisualizationError("Unrecognized node type ({}) in graph for {}".format(item, self.name))
    # TBI Show Dimensions
    name = item.name

    if show_types:
        name = item.name + '\n(' + item.__class__.__name__ + ')'

    if show_dimensions in {ALL, MECHANISMS} and isinstance(item, Mechanism):
        input_str = "in ({})".format(",".join(str(input_port.socket_width)
                                              for input_port in item.input_ports))
        output_str = "out ({})".format(",".join(str(len(np.atleast_1d(output_port.value)))
                                                for output_port in item.output_ports))
        return f"{output_str}\n{name}\n{input_str}"
    if show_dimensions in {ALL, PROJECTIONS} and isinstance(item, Projection):
        # MappingProjections use matrix
        if isinstance(item, MappingProjection):
            value = np.array(item.matrix)
            dim_string = "({})".format("x".join([str(i) for i in value.shape]))
            return "{}\n{}".format(item.name, dim_string)
        # ModulatoryProjections use value
        else:
            value = np.array(item.value)
            dim_string = "({})".format(len(value))
            return "{}\n{}".format(item.name, dim_string)

    if isinstance(item, CompositionInterfaceMechanism):
        name = name.replace('Input_CIM','INPUT')
        name = name.replace('Output_CIM', 'OUTPUT')

    return name


def _animate_execution(composition, active_items, context):
    if composition._component_animation_execution_count is None:
        composition._component_animation_execution_count = 0
    else:
        composition._component_animation_execution_count += 1
    show_graph(
        composition,
        active_items=active_items,
        **composition._animate.addl_show_graph_args,
        output_fmt='gif',
        context=context,
    )


def _generate_gifs(composition, G, active_items, context):

    def create_phase_string(phase):
        return f'%16s' % phase + ' - '

    def create_time_string(time, spec):
        if spec == 'TIME':
            r = time.run
            t = time.trial
            p = time.pass_
            ts = time.time_step
        else:
            r = t = p = ts = '__'
        return f"Time(run: %2s, " % r + f"trial: %2s, " % t + f"pass: %2s, " % p + f"time_step: %2s)" % ts

    G.format = 'gif'
    execution_phase = context.execution_phase
    time = composition.scheduler.get_clock(context).time
    run_num = time.run
    trial_num = time.trial

    if INITIAL_FRAME in active_items:
        phase_string = create_phase_string('Initializing')
        time_string = create_time_string(time, 'BLANKS')

    elif ContextFlags.PROCESSING in execution_phase:
        phase_string = create_phase_string('Processing Phase')
        time_string = create_time_string(time, 'TIME')
    # elif ContextFlags.LEARNING in execution_phase:
    #     time = composition.scheduler_learning.get_clock(context).time
    #     time_string = "Time(run: {}, trial: {}, pass: {}, time_step: {}". \
    #         format(run_num, time.trial, time.pass_, time.time_step)
    #     phase_string = 'Learning Phase - '

    elif ContextFlags.CONTROL in execution_phase:
        phase_string = create_phase_string('Control Phase')
        time_string = create_time_string(time, 'TIME')

    else:
        raise VisualizationError(
            f"PROGRAM ERROR:  Unrecognized phase during execution of {composition.name}: {execution_phase.name}")

    label = f'\n{composition.name}\n{phase_string}{time_string}\n'
    G.attr(label=label)
    G.attr(labelloc='b')
    G.attr(fontname='Monaco')
    G.attr(fontsize='14')
    index = repr(composition._component_animation_execution_count)
    image_filename = '-'.join([repr(run_num), repr(trial_num), index])
    image_file = composition._animate.movie_dir + '/' + image_filename + '.gif'
    G.render(filename=image_filename,
             directory=composition._animate.movie_dir,
             cleanup=True,
             # view=True
             )
    # Append gif to composition._animation
    image = Image.open(image_file)
    # TBI?
    # if not composition._save_images:
    #     remove(image_file)
    if not hasattr(composition, '_animation'):
        composition._animation = [image]
    else:
        composition._animation.append(image)
