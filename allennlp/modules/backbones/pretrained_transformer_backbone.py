from typing import Dict, Optional, Any

from overrides import overrides
import torch

from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.backbones.backbone import Backbone
from allennlp.modules.token_embedders.pretrained_transformer_embedder import (
    PretrainedTransformerEmbedder,
)
from allennlp.nn import util


@Backbone.register("pretrained_transformer")
class PretrainedTransformerBackbone(Backbone):
    """
    Uses a pretrained model from `transformers` as a `Backbone`.

    This class passes most of its arguments to a `PretrainedTransformerEmbedder`, which it uses to
    implement the underlying encoding logic (we duplicate the arguments here instead of taking an
    `Embedder` as a constructor argument just to simplify the user-facing API).

    Registered as a `Backbone` with name "pretrained_transformer".

    # Parameters

    vocab : `Vocabulary`
        Necessary for converting input ids to strings in `make_output_human_readable`.  If you set
        `output_token_strings` to `False`, or if you never call `make_output_human_readable`, then
        this will not be used and can be safely set to `None`.
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
    tokenizer_kwargs: `Dict[str, Any]`, optional (default = `None`)
        Dictionary with
        [additional arguments](https://github.com/huggingface/transformers/blob/155c782a2ccd103cf63ad48a2becd7c76a7d2115/transformers/tokenization_utils.py#L691)
        for `AutoTokenizer.from_pretrained`.
    transformer_kwargs: `Dict[str, Any]`, optional (default = `None`)
        Dictionary with
        [additional arguments](https://github.com/huggingface/transformers/blob/155c782a2ccd103cf63ad48a2becd7c76a7d2115/transformers/modeling_utils.py#L253)
        for `AutoModel.from_pretrained`.
    output_token_strings : `bool`, optional (default = `True`)
        If `True`, we will add the input token ids to the output dictionary in `forward` (with key
        "token_ids"), and convert them to strings in `make_output_human_readable` (with key
        "tokens").  This is necessary for certain demo functionality, and it adds only a trivial
        amount of computation if you are not using a demo.
    vocab_namespace : `str`, optional (default = `"tags"`)
        The namespace to use in conjunction with the `Vocabulary` above.  We use a somewhat
        confusing default of "tags" here, to match what is done in `PretrainedTransformerIndexer`.
    """  # noqa: E501

    def __init__(
        self,
        vocab: Vocabulary,
        model_name: str,
        *,
        max_length: int = None,
        sub_module: str = None,
        train_parameters: bool = True,
        last_layer_only: bool = True,
        override_weights_file: Optional[str] = None,
        override_weights_strip_prefix: Optional[str] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        transformer_kwargs: Optional[Dict[str, Any]] = None,
        output_token_strings: bool = True,
        vocab_namespace: str = "tags",
    ) -> None:
        super().__init__()
        self._vocab = vocab
        self._namespace = vocab_namespace
        self._embedder = PretrainedTransformerEmbedder(
            model_name=model_name,
            max_length=max_length,
            sub_module=sub_module,
            train_parameters=train_parameters,
            last_layer_only=last_layer_only,
            override_weights_file=override_weights_file,
            override_weights_strip_prefix=override_weights_strip_prefix,
            tokenizer_kwargs=tokenizer_kwargs,
            transformer_kwargs=transformer_kwargs,
        )
        self._output_token_strings = output_token_strings

    def forward(self, text: TextFieldTensors) -> Dict[str, torch.Tensor]:  # type: ignore
        if len(text) != 1:
            raise ValueError(
                "PretrainedTransformerBackbone is only compatible with using a single TokenIndexer"
            )
        text_inputs = next(iter(text.values()))
        mask = util.get_text_field_mask(text)
        encoded_text = self._embedder(**text_inputs)
        outputs = {"encoded_text": encoded_text, "encoded_text_mask": mask}
        if self._output_token_strings:
            outputs["token_ids"] = util.get_token_ids_from_text_field_tensors(text)
        return outputs

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if not self._output_token_strings:
            return output_dict

        tokens = []
        for instance_tokens in output_dict["token_ids"]:
            tokens.append(
                [
                    self._vocab.get_token_from_index(token_id.item(), namespace=self._namespace)
                    for token_id in instance_tokens
                ]
            )
        output_dict["tokens"] = tokens
        return output_dict
