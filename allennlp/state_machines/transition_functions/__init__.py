"""
This module contains ``TransitionFunctions`` for state-machine-based decoders.  The
``TransitionFunction`` parameterizes transitions between ``States``.  These ``TransitionFunctions``
are all pytorch `Modules`` that have trainable parameters.  The :class:`BasicTransitionFunction` is
simply an LSTM decoder with attention over an input utterance, and the other classes typically
subclass this and add functionality to it.
"""
# pylint: disable=line-too-long
from allennlp.state_machines.transition_functions.basic_transition_function import BasicTransitionFunction
from allennlp.state_machines.transition_functions.coverage_transition_function import CoverageTransitionFunction
from allennlp.state_machines.transition_functions.linking_coverage_transition_function import LinkingCoverageTransitionFunction
from allennlp.state_machines.transition_functions.linking_transition_function import LinkingTransitionFunction
from allennlp.state_machines.transition_functions.transition_function import TransitionFunction
