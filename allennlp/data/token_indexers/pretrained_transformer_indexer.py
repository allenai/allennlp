from typing import Dict, List
import logging

from overrides import overrides
from transformers.tokenization_auto import AutoTokenizer
import torch

from allennlp.common.util import pad_sequence_to_length
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer

logger = logging.getLogger(__name__)


@TokenIndexer.register("pretrained_transformer")
class PretrainedTransformerIndexer(TokenIndexer[int]):
    """
    If you are using this token indexer, you should also add predefined transformers vocab
    to our ``Vocabulary`` object. Concretly, there is a ``pretrained_transformers_vocab``
    constructor argument of the ``Vocabulary`` that you should set.

    This ``TokenIndexer`` does not extend the vocabulary and assumes Tokens already have ```text_id```
    property set for them.  We still require ``model_name`` however to handle padding correcty.

    Parameters
    ----------
    model_name : ``str``
        The name of the ``transformers`` model to use.
    """

    def __init__(self, model_name: str, token_min_padding_length: int = 0) -> None:
        super().__init__(token_min_padding_length)
        # we still need to get proper padding value..
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._padding_value = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
        logger.info(f"Using token indexer padding value of {self._padding_value}")

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        # If we only use pretrained models, we don't need to do anything here.
        pass

    @overrides
    def tokens_to_indices(
        self, tokens: List[Token], vocabulary: Vocabulary, index_name: str
    ) -> Dict[str, List[int]]:
        indices: List[int] = []
        for token in tokens:
            if getattr(token, "text_id", None) is not None:
                # `text_id` being set on the token means that we aren't using the vocab, we just use
                # this id instead. Id comes from the pretrained vocab.
                # # It computed in PretrainedTransformerTokenizer.
                indices.append(token.text_id)
            else:
                raise KeyError("Field text_id is not set for the following token: " + token.text)

        return {index_name: indices}

    @overrides
    def get_padding_lengths(self, token: int) -> Dict[str, int]:
        return {}

    @overrides
    def as_padded_tensor(
        self,
        tokens: Dict[str, List[int]],
        desired_num_tokens: Dict[str, int],
        padding_lengths: Dict[str, int],
    ) -> Dict[str, torch.Tensor]:
        return {
            key: torch.LongTensor(
                pad_sequence_to_length(
                    val, desired_num_tokens[key], default_value=lambda: self._padding_value
                )
            )
            for key, val in tokens.items()
        }

    def __eq__(self, other):
        if isinstance(other, PretrainedTransformerIndexer):
            for key in self.__dict__:
                if key == "tokenizer":
                    # This is a reference to a function in the huggingface code, which we can't
                    # really modify to make this clean.  So we special-case it.
                    continue
                if self.__dict__[key] != other.__dict__[key]:
                    return False
            return True
        return NotImplemented
