from typing import Dict, Optional

import torch

from allennlp.data import TextFieldTensors
from allennlp.modules.backbones.backbone import Backbone
from allennlp.modules.token_embedders.pretrained_transformer_embedder import PretrainedTransformerEmbedder
from allennlp.nn import util


@Backbone.register("pretrained_transformer")
class PretrainedTransformerBackbone(Backbone):
    """
    Uses a pretrained model from `transformers` as a `Backbone`.

    This class passes all of its arguments to a `PretrainedTransformerEmbedder`, which it uses to
    implement the underlying encoding logic (we duplicate the arguments here instead of taking an
    `Embedder` as a constructor argument just to simplify the user-facing API).

    Registered as a `Backbone` with name "pretrained_transformer".

    # Parameters

    model_name : `str`
        The name of the `transformers` model to use. Should be the same as the corresponding
        `PretrainedTransformerIndexer`.
    max_length : `int`, optional (default = `None`)
        If positive, folds input token IDs into multiple segments of this length, pass them
        through the transformer model independently, and concatenate the final representations.
        Should be set to the same value as the `max_length` option on the
        `PretrainedTransformerIndexer`.
    sub_module: `str`, optional (default = `None`)
        The name of a submodule of the transformer to be used as the embedder. Some transformers naturally act
        as embedders such as BERT. However, other models consist of encoder and decoder, in which case we just
        want to use the encoder.
    train_parameters: `bool`, optional (default = `True`)
        If this is `True`, the transformer weights get updated during training.
    last_layer_only: `bool`, optional (default = `True`)
        When `True` (the default), only the final layer of the pretrained transformer is taken
        for the embeddings. But if set to `False`, a scalar mix of all of the layers
        is used.
    """

    def __init__(
        self,
        model_name: str,
        *,
        max_length: int = None,
        sub_module: str = None,
        train_parameters: bool = True,
        last_layer_only: bool = True,
        override_weights_file: Optional[str] = None,
        override_weights_strip_prefix: Optional[str] = None
    ) -> None:
        super().__init__()
        self._embedder = PretrainedTransformerEmbedder(
            model_name=model_name,
            max_length=max_length,
            sub_module=sub_module,
            train_parameters=train_parameters,
            last_layer_only=last_layer_only,
            override_weights_file=override_weights_file,
            override_weights_strip_prefix=override_weights_strip_prefix,
        )

    def forward(self, text: TextFieldTensors) -> Dict[str, torch.Tensor]:
        if len(text) != 1:
            raise ValueError(
                "PretrainedTransformerBackbone is only compatible with using a single TokenIndexer"
            )
        text_inputs = next(iter(text.values()))
        mask = util.get_text_field_mask(text)
        encoded_text = self._embedder(**text_inputs)
        return {"encoded_text": encoded_text, "encoded_text_mask": mask}
