# The AllenNLP Semantic Parsing Framework

_Semantic parsing_ is the task of mapping language to some kind of formal meaning representation.
This meaning representation could be a logical statement in lambda calculus or lambda-DCS, a set of
instructions for a robot to follow, or even a python, java, or SQL program.  In some cases these
meaning representations are directly executable in some environment (like executing a query against
a database, or running python code), and in others they are simply an attempt to normalize the
semantics of a natural language utterance (like Abstract Meaning Representations or open-domain CCG
semantics).  The thing all of these variations have in common is that they try to capture the
meaning of language in some form, and there typically isn't a direct mapping from words in the
utterance to the pieces of the meaning representation.  (TODO: add links above)

We approach this problem in AllenNLP with encoder-decoder models, similar to the seq2seq models
used in neural machine translation or summarization.  That is, we encode the input utterance (and
whatever other context is available) using some neural encoder, then use a decoder to produce a
statement in the output language. Because the output language in semantic parsing is a formal
language, however, we use _constrained decoding_, forcing the model to only produce valid meaning
representations.

We accomplish this with a generic state machine decoder, where the model defines a transition
system stipulating which actions are valid at any particular state.  If the "actions" are all of
the tokens in your output vocabulary, and all actions are valid at every state, this would be a
standard seq2seq model (FOOTNOTE: though a much less efficient one than other ways of doing this).
By changing what the actions represent, or constraining the valid actions at each timestep of
decoding, however, we can learn better models that are tailored for our task.  Note also that state
machines are more general than semantic parsers, and you can use this framework for any structured
prediction problem that you can formulate as a transition system.

The rest of this tutorial will walk you through the pieces of the semantic parsing framework (TODO:
make these links):

- Transition functions and training algorithms
- Keeping track of the decoder's state
- Defining a transition system to specify what actions there are and when they can be taken
- Adding context to your model (like a table or a database)
- Defining an execution engine
- Putting the context, the transition system, and the execution engine together into a `World`

## Training a transition function

The fundamental piece of the semantic parsing framework is the `TransitionFunction`.  This is a
pytorch `Module` that parameterizes state transitions.  That is, given a `State`, the
`TransitionFunction` returns a scored and ranked list of next `States`:

```python
class TransitionFunction(torch.nn.Module, Generic[StateType]):
    def take_step(self,
                  state: StateType,
                  max_actions: int = None,
                  allowed_actions: List[Set] = None) -> List[StateType]:
        raise NotImplementedError
```

If you ignore the optional arguments for a minute, this is a simple function that takes a `State`
of a particular type and returns a list of `States` of the same type.  This function will have some
parameters, probably based on some kind of LSTM decoder hidden state, that are used to score each
valid action, construct new `States` for those actions, and return them.  The optional parameters
are for efficiency, so you don't construct huge numbers of `State` objects if you know you're doing
a beam search with size `k`, or if you're training and only need to score a few select action
sequences.

The `TransitionFunction` (along with a `State` object which we'll get to later TODO: add link)
forms the basis of our decoder state machine, and now we need some way to train it.  There are a
lot of ways to train state machines, depending on what kind of supervision you have.  You could
have fully-labeled correct action sequences, a set of possibly correct action sequences, a way to
check the correctness of finished states, or just a reward function on finished or intermediate
states.  To handle the variety of supervision signals that could be used, we provide a simple
`DecoderTrainer` interface that's generic over the supervision type:

```python
class DecoderTrainer(Generic[SupervisionType]):
    def decode(self,
               initial_state: State,
               transition_function: TransitionFunction,
               supervision: SupervisionType) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
```

The `SupervisionType` here could be a correct action sequence or a reward function (or anything
else you choose); the generic interface makes it so that you can easily swap out trainers that take
the same supervision signal (such as maximum likelihood with beam search optimization, or various
reinforcement learning algorithms with each other).  The `DecoderTrainer` requires an initial
`State`, a `TransitionFunction`, and a supervision signal, and it returns a `loss` tensor that we
use to compute gradients and train the `TransitionFunction` and the encoder that created the
initial `State` (the return value is a dictionary so you can return whatever else you want, too).

There are several examples of `DecoderTrainers` and `TransitionFunctions` already in the library,
and you can create your own if what we have doesn't fit your needs (and if you implement your own
that you think would be useful for others, please consider contributing it back!).  Here are a few
examples (not an exhaustive list): TODO: add links

`DecoderTrainers`:
- MaximumMarginalLikelihood (when you have a set of possibly correct action sequences)
- EmpiricalRiskMinimization (when you have a reward function)

`TransitionFunctions`:
- `BasicTransitionFunction`: a simple LSTM decoder with attention that uses a grammar to constrain
  the actions available at each state.
- `LinkingTransitionFunction`: an extension of the `BasicTransitionFunction` that allows some
  actions to be parameterized by linking to words in the utterance, instead of having an embedding
for each action.  This allows for predicting actions at test time that were never seen at training
time, to do various kinds of zero-shot prediction.


## Tracking the State of the decoder

Our `TransitionFunctions` operate on `States`, scoring actions available at each `State` and
returning ranked lists of new ones.  In order for this to work, we need some way of representing
that `State`, including any intermediate computations that are necessary for scoring actions.  We
do this with a simple `State` object:

```python
class State:
    def __init__(self,
                 batch_indices: List[int],
                 action_history: List[List[int]],
                 score: List[torch.Tensor]) -> None:
        ...

    def is_finished(self) -> bool:
        raise NotImplementedError

    @classmethod
    def combine_states(cls, states: List[State]) -> State:
        raise NotImplementedError
```

The first thing to notice here is that we're dealing with lists - in order to speed things up a
bit, we batch some of our computation together.  A semantic parsing `Model` will typically encode
its (batched) inputs and construct a (batched) `State`, where each instance in the batch gets an
entry in the initial `State`, with an empty action history and a score of 0.  During decoding, if
you are doing any kind of search, we can further group together computation from multiple `States`
on the beam for the same batch instance, which lets us, for example, only run the decoder LSTM cell
on the GPU once per timestep.  We do this with the `combine_states` method, which might result in a
`State` that has multiple entries per batch index.

But what goes into a `State`?  The generic base class just has the minimal interface necessary for
the `DecoderTrainer` to interact with it: a score, what actions have been taken, a way to tell if
we should stop, and a way to combine states together for better batching.  Your
`TransitionFunction` will certainly need more information, like the actual hidden state of a
decoder LSTM, or the actions that are available in the current state.  To group together some
common pieces of this internal state, we have a few `Statelet` classes available.  We'll go over
two of them: `RnnStatelet` and `GrammarStatelet`.

`RnnStatelet` keeps track of the internal state of a decoder RNN:

```python
class RnnStatelet:
    def __init__(self,
                 hidden_state: torch.Tensor,
                 memory_cell: torch.Tensor,
                 previous_action_embedding: torch.Tensor,
                 attended_input: torch.Tensor,
                 encoder_outputs: List[torch.Tensor],
                 encoder_output_mask: List[torch.Tensor]) -> None:
        ...
```

This includes the typical hidden state and memory cell for an LSTM, but also the embedding for the
action taken at the previous timestep (which will be used as input to the next timestep), and the
previous attended input (which is also used as part of the input to the next timestep, for
"attention feeding").  And because we compute attention over the encoded input representations in
the LSTM decoder, we also include those inputs (and their mask) as part of the `RnnStatelet`.

`GrammarStatelet` keeps track of the current state of the grammar during decoding: what
non-terminal is at the top of the non-terminal stack, what actions are available, and so on:

```python
class GrammarStatelet:
    def __init__(self,
                 nonterminal_stack: List[str],
                 valid_actions: Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor, List[int]]]],
                 is_nonterminal: Callable[[str], bool]) -> None:
        ...

    def is_finished(self) -> bool:
        ...

    def get_valid_actions(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, List[int]]]:
        ...

    def take_action(self, production_rule: str) -> 'GrammarStatelet':
        ...
```
