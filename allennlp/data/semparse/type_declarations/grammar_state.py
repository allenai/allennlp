class GrammarState:
    """
    A ``GrammarState`` specifies the currently valid actions at every step of decoding.

    If we had a global, context-free grammar, this would not be necessary - the currently valid
    actions would always be the same, and we would not need to represent the current state.
    However, our grammar is not context free (we have lambda expressions that introduce
    context-dependent production rules), and it is not global (each instance can have its own
    entities of a particular type, or even its own functions).

    We thus recognize three different sources of valid actions, and we treat them separately.  The
    first are actions that come from the type declaration; these are computed once by the model and
    shared across all ``GrammarStates`` produced by that model.  The second are actions that come
    from the current instance; these are computed for each instance in ``model.forward``, and are
    shared across all decoding states for that instance.  The last are actions that come from the
    current state of the decoder; these are updated after every action taken by the decoder, though
    only some actions initiate changes.

    In practice, though, we use the ``World`` class to get the first two sources of valid actions
    at the same time, and let ``World`` deal with only computing the global actions once.  There is
    one ``World`` for each instance.
    """
