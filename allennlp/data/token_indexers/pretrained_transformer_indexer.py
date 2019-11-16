from typing import Dict, List
import logging

from overrides import overrides
from transformers.tokenization_auto import AutoTokenizer
import torch

from allennlp.common.util import pad_sequence_to_length
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.token_indexers import SingleIdTokenIndexer

logger = logging.getLogger(__name__)


@TokenIndexer.register("pretrained_transformer")
class PretrainedTransformerIndexer(SingleIdTokenIndexer):
    """
    This ``TokenIndexer`` assumes that Tokens already have their indexes in them.
    We still require ``model_name`` because we want to form allennlp vocabulary from pretrained one.
    This ``Indexer`` is only really appropriate to use if you've also used a
    corresponding :class:`PretrainedTransformerTokenizer` to tokenize your input.  Otherwise you'll
    have a mismatch between your tokens and your vocabulary, and you'll get a lot of UNK tokens.

    Parameters
    ----------
    model_name : ``str``
        The name of the ``transformers`` model to use.
    namespace : ``str``, optional (default=``tags``)
        We will add the tokens in the pytorch_transformer vocabulary to this vocabulary namespace.
        We use a somewhat confusing default value of ``tags`` so that we do not add padding or UNK
        tokens to this namespace, which would break on loading because we wouldn't find our default
        OOV token.
    """

    def __init__(
        self, model_name: str, namespace: str = "tags", token_min_padding_length: int = 0
    ) -> None:
        super().__init__(
            namespace=namespace,
            lowercase_tokens=False,
            start_tokens=None,
            end_tokens=None,
            token_min_padding_length=token_min_padding_length,
        )
        self._model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._namespace = namespace
        self._added_to_vocabulary = False
        self._padding_value = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]
        logger.info(f"Using token indexer padding value of {self._padding_value}")

    def _add_encoding_to_vocabulary(self, vocabulary: Vocabulary, vocab_field_name: str) -> None:
        pretrained_vocab = getattr(self.tokenizer, vocab_field_name)
        for word, idx in pretrained_vocab.items():
            vocabulary._token_to_index[self._namespace][word] = idx
            vocabulary._index_to_token[self._namespace][idx] = word

    def _maybe_add_transformers_vocab_to_allennlp(self, vocabulary: Vocabulary):
        if not self._added_to_vocabulary:
            if hasattr(self.tokenizer, "vocab"):
                self._add_encoding_to_vocabulary(vocabulary, "vocab")
                self._added_to_vocabulary = True
            elif hasattr(self.tokenizer, "encoder"):
                self._add_encoding_to_vocabulary(vocabulary, "encoder")
                self._added_to_vocabulary = True
            else:
                logger.warning(
                    """Wasn't able to fetch pretrained vocabulary.
                       Your tokens will still be correctly indexed, but vocabulary file will not be saved."""
                )
                self._added_to_vocabulary = True

    @overrides
    def tokens_to_indices(
        self, tokens: List[Token], vocabulary: Vocabulary, index_name: str
    ) -> Dict[str, List[int]]:
        self._maybe_add_transformers_vocab_to_allennlp(vocabulary)

        indices: List[int] = []
        for token in tokens:
            print(getattr(token, "text_id", None) is not None, token)

            if getattr(token, "text_id", None) is not None:
                # print(getattr(token, "text_id", None) is not None)
                # print(token)
                # `text_id` being set on the token means that we aren't using the vocab, we just use
                # this id instead. Id comes from the pretrained vocab.
                # # It computed in PretrainedTransformerTokenizer.
                indices.append(token.text_id)
            else:
                raise KeyError("Field text_id is not set for the following token: " + token.text)

        return {index_name: indices}

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
