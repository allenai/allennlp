import torch
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.data import Vocabulary
from allennlp.common import Params
from allennlp.nn.util import get_text_field_mask


@TokenEmbedder.register("bow_token_embedder")
class BagOfWordsTokenEmbedder(TokenEmbedder):
    """
    Embeds a document in a bag of words representation. Each document is represented by
    a vector of length vocabulary size. Each cell corresponds to a different token in
    the vocabulary, and is populated with the number of times that token appears
    in the document. This embedding serves as a strong baseline for document-level
    representation learning.

    Parameters
    ----------
    vocab_size: int
        size of vocabulary
    projection_dim : int, optional (default = None)
        if specified, will project the resulting bag of words representation
        to specified dimension.
    """
    def __init__(self, vocab_size: int, projection_dim: int = None) -> None:
        super(BagOfWordsTokenEmbedder, self).__init__()
        self.vocab_size = vocab_size
        if projection_dim:
            self._projection = torch.nn.Linear(vocab_size, projection_dim)
        else:
            self._projection = None
        self.output_dim = projection_dim or vocab_size

    def get_output_dim(self):
        return self.num_embeddings

    def compute_bow(self, tokens: torch.IntTensor) -> torch.Tensor:
        """
        Compute a bag of words representation with size batch_size x vocab_size

        Parameters
        ----------
        tokens : ``Dict[str, torch.Tensor]``
            tokens to compute bag of words representation of
        """
        bow_vectors = []
        mask = get_text_field_mask({'tokens': tokens})
        for document, doc_mask in zip(tokens, mask):
            document = torch.masked_select(document, doc_mask.byte())
            vec = torch.bincount(document, minlength=self.vocab_size).float()
            vec = vec.view(1, -1)
            bow_vectors.append(vec)
        return torch.cat(bow_vectors, 0)

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
        bow_output = self.compute_bow(inputs)
        if self._projection:
            projection = self._projection
            bow_output = projection(bow_output)
        return bow_output

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'BagOfWordsTokenEmbedder':  # type: ignore
        # pylint: disable=arguments-differ
        """
        we look for a ``vocab_namespace`` key in the parameter dictionary
        to know which vocabulary to use.
        """
        vocab_namespace = params.pop("vocab_namespace", "tokens")
        vocab_size = vocab.get_vocab_size(vocab_namespace)
        projection_dim = params.pop_int("projection_dim", None)
        params.assert_empty(cls.__name__)
        return cls(vocab_size=vocab_size,
                   projection_dim=projection_dim)
