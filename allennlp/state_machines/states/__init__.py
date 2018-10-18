"""
This module contains the ``State`` abstraction for defining state-machine-based decoders, and some
pre-built concrete ``State`` classes for various kinds of decoding (e.g., a ``GrammarBasedState``
for doing grammar-based decoding, where the output is a sequence of production rules from a
grammar).

The module also has some ``Statelet`` classes to help represent the ``State`` by grouping together
related pieces, including ``RnnStatelet``, which you can use to keep track of a decoder RNN's
internal state, ``GrammarStatelet``, which keeps track of what actions are allowed at each timestep
of decoding (if your outputs are production rules from a grammar), and ``ChecklistStatelet`` that
keeps track of coverage information if you are training a coverage-based parser.
"""
from allennlp.state_machines.states.checklist_statelet import ChecklistStatelet
from allennlp.state_machines.states.coverage_state import CoverageState
from allennlp.state_machines.states.grammar_based_state import GrammarBasedState
from allennlp.state_machines.states.grammar_statelet import GrammarStatelet
from allennlp.state_machines.states.lambda_grammar_statelet import LambdaGrammarStatelet
from allennlp.state_machines.states.rnn_statelet import RnnStatelet
from allennlp.state_machines.states.state import State
