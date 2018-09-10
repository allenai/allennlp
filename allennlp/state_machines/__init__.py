"""
This module contains code for using state machines in a model to do transition-based decoding.
"Transition-based decoding" is where you start in some state, iteratively transition between
states, and have some kind of supervision signal that tells you which end states, or which
transition sequences, are "good".

Typical seq2seq decoding, where you have a fixed vocabulary and no constraints on your output, can
be done much more efficiently than we do in this code.  This is intended for structured models that
have constraints on their outputs.

The key abstractions in this code are the following:

    - ``State`` represents the current state of decoding, containing a list of all of the actions
      taken so far, and a current score for the state.  It also has methods around determining
      whether the state is "finished" and for combining states for batched computation.
    - ``TransitionFunction`` is a ``torch.nn.Module`` that models the transition function between
      states.  Its main method is ``take_step``, which generates a ranked list of next states given
      a current state.
    - ``DecoderTrainer`` is an algorithm for training the transition function with some kind of
      supervision signal.  There are many options for training algorithms and supervision signals;
      this is an abstract class that is generic over the type of the supervision signal.

There is also a generic ``BeamSearch`` class for finding the ``k`` highest-scoring transition
sequences given a trained ``TransitionFunction`` and an initial ``State``.
"""
from allennlp.state_machines.beam_search import BeamSearch
from allennlp.state_machines.constrained_beam_search import ConstrainedBeamSearch
from allennlp.state_machines.states import State
from allennlp.state_machines.trainers import DecoderTrainer
from allennlp.state_machines.transition_functions import TransitionFunction
