import torch
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.data import Vocabulary
from allennlp.common import Params
from allennlp.nn.util import get_text_field_mask
from allennlp.common.checks import ConfigurationError


@TokenEmbedder.register("bag_of_word_counts")
class BagOfWordCountsTokenEmbedder(TokenEmbedder):
    """
    Represents a sequence of tokens as a bag of (discrete) word ids, as it was done
    in the pre-neural days.

    Each sequence gets a vector of length vocabulary size, where the i'th entry in the vector
    corresponds to number of times the i'th token in the vocabulary appears in the sequence.

    By default, we ignore padding tokens.

    Parameters
    ----------
    vocab: ``Vocabulary``
    vocab_namespace: ``str``
        namespace of vocabulary to embed
    projection_dim : ``int``, optional (default = ``None``)
        if specified, will project the resulting bag of words representation
        to specified dimension.
    ignore_oov : ``bool``, optional (default = ``False``)
        If true, we ignore the OOV token.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 vocab_namespace: str,
                 projection_dim: int = None,
                 ignore_oov: bool = False) -> None:
        super().__init__()
        self.vocab = vocab
        self.vocab_size = vocab.get_vocab_size(vocab_namespace)
        if projection_dim:
            self._projection = torch.nn.Linear(self.vocab_size, projection_dim)
        else:
            self._projection = None
        self._ignore_oov = ignore_oov
        oov_token = vocab._oov_token  # pylint: disable=protected-access
        self._oov_idx = vocab.get_token_to_index_vocabulary(vocab_namespace).get(oov_token)
        if self._oov_idx is None:
            raise ConfigurationError("OOV token does not exist in vocabulary namespace {}".format(vocab_namespace))
        self.output_dim = projection_dim or self.vocab_size

    def get_output_dim(self):
        return self.output_dim

    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs: ``torch.Tensor``
            Shape ``(batch_size, timesteps, sequence_length)`` of word ids
            representing the current batch.

        Returns
        -------
        The bag-of-words representations for the input sequence, shape
        ``(batch_size, vocab_size)``
        """
        bag_of_words_vectors = []

        mask = get_text_field_mask({'tokens': inputs})
        if self._ignore_oov:
            # also mask out positions corresponding to oov
            mask *= (inputs != self._oov_idx).long()
        for document, doc_mask in zip(inputs, mask):
            document = torch.masked_select(document, doc_mask.to(dtype=torch.bool))
            vec = torch.bincount(document, minlength=self.vocab_size).float()
            vec = vec.view(1, -1)
            bag_of_words_vectors.append(vec)
        bag_of_words_output = torch.cat(bag_of_words_vectors, 0)

        if self._projection:
            projection = self._projection
            bag_of_words_output = projection(bag_of_words_output)
        return bag_of_words_output

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'BagOfWordCountsTokenEmbedder':  # type: ignore
        # pylint: disable=arguments-differ
        """
        we look for a ``vocab_namespace`` key in the parameter dictionary
        to know which vocabulary to use.
        """

        vocab_namespace = params.pop("vocab_namespace", "tokens")
        projection_dim = params.pop_int("projection_dim", None)
        ignore_oov = params.pop_bool("ignore_oov", False)
        params.assert_empty(cls.__name__)
        return cls(vocab=vocab,
                   vocab_namespace=vocab_namespace,
                   ignore_oov=ignore_oov,
                   projection_dim=projection_dim)
