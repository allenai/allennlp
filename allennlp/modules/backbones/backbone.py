from typing import Dict

import torch

from allennlp.common import Registrable

class Backbone(Registrable, torch.nn.Module):
    """
    A `Backbone` operates on basic model inputs and produces some encoding of those inputs that will
    be shared among one or more `Heads` in a multi-task setting.  For plain text inputs, this is
    often a transformer.

    This class does not currently specify any API; it exists to give us a `Registrable` class that
    we can use as a type annotation on `Model` classes that want to use a backbone.  The expectation
    is that this will take the same inputs as a typical model, but return intermediate
    representations.  These should generally be returned as a dictionary, from which the caller will
    have to pull out what they want and use as desired.  As a convention that these modules should
    generally follow, their outputs should have the same name as the given input, prepended with
    `encoded_`.  So, a backbone that encodes a `text` input should return an output called
    `encoded_text`.  This convention allows easier exchangeability of these backbone modules.

    Additionally, as downstream `Heads` will typically need mask information, but after encoding
    have no way of computing it, a `Backbone` should also return a mask for each of its outputs,
    with the same name as the output but with `_mask` appended.  So in our example of `text` as
    input, the output should have an entry called `encoded_text_mask`.
    """

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
