from typing import Dict, List
import itertools

from overrides import overrides
import pytorch_transformers
from pytorch_transformers.tokenization_utils import PreTrainedTokenizer
import torch

from allennlp.common.util import pad_sequence_to_length
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer


@TokenIndexer.register("pretrained_transformer")
class PretrainedTransformerIndexer(TokenIndexer[int]):
    """
    This :class:`TokenIndexer` uses a tokenizer from the ``pytorch_transformers`` repository to
    index tokens.  This ``Indexer`` is only really appropriate to use if you've also used a
    corresponding :class:`PretrainedTransformerTokenizer` to tokenize your input.  Otherwise you'll
    have a mismatch between your tokens and your vocabulary, and you'll get a lot of UNK tokens.

    Parameters
    ----------
    model_name : ``str``
        The name of the ``pytorch_transformers`` model to use.
    do_lowercase : ``str``
        Whether to lowercase the tokens (this should match the casing of the model name that you
        pass)
    namespace : ``str``, optional (default=``tags``)
        We will add the tokens in the pytorch_transformer vocabulary to this vocabulary namespace.
        We use a somewhat confusing default value of ``tags`` so that we do not add padding or UNK
        tokens to this namespace, which would break on loading because we wouldn't find our default
        OOV token.
    """
    # pylint: disable=no-self-use
    def __init__(self,
                 model_name: str,
                 do_lowercase: bool,
                 namespace: str = "tags",
                 token_min_padding_length: int = 0) -> None:
        super().__init__(token_min_padding_length)
        if model_name.endswith("-cased") and do_lowercase:
            logger.warning("Your pretrained model appears to be cased, "
                           "but your indexer is lowercasing tokens.")
        elif model_name.endswith("-uncased") and not do_lowercase:
            logger.warning("Your pretrained model appears to be uncased, "
                           "but your indexer is not lowercasing tokens.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=do_lowercase)
        self._namespace = namespace
        self._added_to_vocabulary = False

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        # If we only use pretrained models, we don't need to do anything here.
        pass

    def _add_encoding_to_vocabulary(self, vocabulary: Vocabulary) -> None:
        # pylint: disable=protected-access
        for word, idx in self.tokenizer.vocab.items():
            vocabulary._token_to_index[self._namespace][word] = idx
            vocabulary._index_to_token[self._namespace][idx] = word

    @overrides
    def tokens_to_indices(self,
                          tokens: List[Token],
                          vocabulary: Vocabulary,
                          index_name: str) -> Dict[str, List[int]]:
        if not self._added_to_vocabulary:
            self._add_encoding_to_vocabulary(vocabulary)
            self._added_to_vocabulary = True
        token_text = [token.text for token in tokens]
        indices = self.tokenizer.convert_tokens_to_ids(token_text)

        return {index_name: indices}

    @overrides
    def get_padding_lengths(self, token: int) -> Dict[str, int]:  # pylint: disable=unused-argument
        return {}

    @overrides
    def as_padded_tensor(self,
                         tokens: Dict[str, List[int]],
                         desired_num_tokens: Dict[str, int],
                         padding_lengths: Dict[str, int]) -> Dict[str, torch.Tensor]:  # pylint: disable=unused-argument
        return {key: torch.LongTensor(pad_sequence_to_length(val, desired_num_tokens[key]))
                for key, val in tokens.items()}
