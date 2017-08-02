from typing import Dict

import torch
from overrides import overrides

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder


@TextFieldEmbedder.register("model")
class ModelTextFieldEmbedder(TextFieldEmbedder):
    """
    This is a ``TextFieldEmbedder`` that wraps a ``Model``.  The model must accept a single text
    field as input.  We run ``forward`` on the model, then pull out one of the values in its output
    dictionary to return.

    NOTE: This is still largely experimental and needs some real-world testing before I'm confident
    this API is correct and actually works well.

    Parameters
    ----------
    model : ``Model``
        The model that we will use to embed the text field.
    model_input_name : ``str``
        The name the model assigns to the text field input we want to use.  For example, the
        :class:`~allennlp.models.SimpleTagger` model takes a text input field named ``tokens`` -
        that's the name that should be passed here.
    model_output_name : ``str``
        The key that we will look for in the model's output dictionary.  The model must have
        included a token-level representation in its output dictionary for it to be useful here.
    model_output_dim : ``int``
        ``TextFieldEmbedders`` need to implement ``get_output_dim()``.  We could either make the
        model implement some method that tells us what the output dim is, or we can take it as a
        parameter.  We're opting for the second option here.
    """
    def __init__(self,
                 model: Model,
                 model_input_name: str,
                 model_output_name: str,
                 model_output_dim: int) -> None:
        super(ModelTextFieldEmbedder, self).__init__()
        self._model = model
        self._input_name = model_input_name
        self._output_name = model_output_name
        self._output_dim = model_output_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(self, text_field_input: Dict[str, torch.Tensor]) -> torch.Tensor:
        arg_dict = {self._input_name: text_field_input}

        # TODO(mattg): switch to using __call__ once pytorch supports dictionary return values.
        return self._model.forward(**arg_dict)[self._output_name]

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'ModelTextFieldEmbedder':
        # TODO(mattg): we probably need to load the saved model here somehow, and have some
        # parameters around whether this model should be trainable.  Not even sure how to set this
        # after-the-fact with pytorch.
        model = Model.from_params(vocab, params.pop('model'))
        model_input_name = params.pop('input_name')
        model_output_name = params.pop('output_name')
        model_output_dim = params.pop('output_dim')
        params.assert_empty(cls.__name__)
        return cls(model=model,
                   model_input_name=model_input_name,
                   model_output_name=model_output_name,
                   model_output_dim=model_output_dim)
