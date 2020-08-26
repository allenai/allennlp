from allennlp.models.model import Model


class Head(Model):
    """
    A `Head` is a `Model` that takes _already encoded input_ and typically does simple computation
    before returning a loss.

    There isn't currently any difference in API between a `Model` and a `Head`, but we have this
    separate type as both a signaling mechanism for what to expect when looking at a `Head` class,
    and so that we can use this as a more informative type annotation when building models that use
    `Heads` as inputs.
    """

    pass
