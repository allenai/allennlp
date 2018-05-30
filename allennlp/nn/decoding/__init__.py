"""
This module contains code for transition-based decoding.  "Transition-based decoding" is where you
start in some state, iteratively transition between states, and have some kind of supervision
signal that tells you which end states, or which transition sequences, are "good".

If you want to do decoding for a vocabulary-based model, where the allowable outputs are the same
at every timestep of decoding, this code is not what you are looking for, and it will be quite
inefficient compared to other things you could do.

The key abstractions in this code are the following:

    - ``DecoderState`` represents the current state of decoding, containing a list of all of the
      actions taken so far, and a current score for the state.  It also has methods around
      determining whether the state is "finished" and for combining states for batched computation.
    - ``DecoderStep`` is a ``torch.nn.Module`` that models the transition function between states.
      Its main method is ``take_step``, which generates a ranked list of next states given a
      current state.
    - ``DecoderTrainer`` is an algorithm for training the transition function with some kind of
      supervision signal.  There are many options for training algorithms and supervision signals;
      this is an abstract class that is generic over the type of the supervision signal.

The module also has some classes to help represent the ``DecoderState``, including ``RnnState``,
which you can use to keep track of a decoder RNN's internal state, ``GrammarState``, which
keeps track of what actions are allowed at each timestep of decoding, if your outputs are
production rules from a grammar, and ``ChecklistState`` that keeps track of coverage inforation if
you are training a coverage based parser.

There is also a generic ``BeamSearch`` class for finding the ``k`` highest-scoring transition
sequences given a trained ``DecoderStep`` and an initial ``DecoderState``.
"""
from allennlp.nn.decoding.beam_search import BeamSearch
from allennlp.nn.decoding.checklist_state import ChecklistState
from allennlp.nn.decoding.constrained_beam_search import ConstrainedBeamSearch
from allennlp.nn.decoding.decoder_state import DecoderState
from allennlp.nn.decoding.decoder_step import DecoderStep
from allennlp.nn.decoding.decoder_trainers.decoder_trainer import DecoderTrainer
from allennlp.nn.decoding.grammar_state import GrammarState
from allennlp.nn.decoding.rnn_state import RnnState
