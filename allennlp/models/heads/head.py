from allennlp.models.model import Model


class Head(Model):
    """
    A `Head` is a `Model` that takes _already encoded input_ and typically does simple computation
    before returning a loss.

    There isn't currently any difference in API between a `Model` and a `Head`, but we have this
    separate type as both a signaling mechanism for what to expect when looking at a `Head` class,
    and so that we can use this as a more informative type annotation when building models that use
    `Heads` as inputs.

    One additional consideration in a `Head` is that `make_output_human_readable` needs to account
    for the case where it gets called without first having `forward` be called on the head.  This is
    because at the point where we call `make_output_human_readable`, we don't know which heads were
    used in `forward`, and trying to save the state is messy.  So just make sure that you always
    have conditional logic in `make_output_human_readable` when you implement a `Head`.
    """

    pass
