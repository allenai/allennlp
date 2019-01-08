from typing import Union

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.language_model import LanguageModel
from allennlp.models.model import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.nn import InitializerApplicator


@Model.register('bidirectional-language-model')
@Model.register('bidirectional_language_model')
class BidirectionalLanguageModel(LanguageModel):
    """
    The ``BidirectionalLanguageModel`` applies a bidirectional "contextualizing"
    ``Seq2SeqEncoder`` to uncontextualized embeddings, using a ``SoftmaxLoss``
    module (defined above) to compute the language modeling loss.

    It is IMPORTANT that your bidirectional ``Seq2SeqEncoder`` does not do any
    "peeking ahead". That is, for its forward direction it should only consider
    embeddings at previous timesteps, and for its backward direction only embeddings
    at subsequent timesteps. If this condition is not met, your language model is
    cheating.

    Parameters
    ----------
    vocab: ``Vocabulary``
    text_field_embedder: ``TextFieldEmbedder``
        Used to embed the indexed tokens we get in ``forward``.
    contextualizer: ``Seq2SeqEncoder``
        Used to "contextualize" the embeddings. As described above,
        this encoder must not cheat by peeking ahead.
    dropout: ``float``, optional (default: None)
        If specified, dropout is applied to the contextualized embeddings before computation of
        the softmax. The contextualized embeddings themselves are returned without dropout.
    loss_scale: ``Union[float, str]``, optional (default: 1.0)
        This scaling factor is applied to the average language model loss.
        You can also specify ``"n_samples"`` in which case we compute total
        loss across all predictions.
    num_samples: ``int``, optional (default: None)
        If provided, the model will use ``SampledSoftmaxLoss``
        with the specified number of samples. Otherwise, it will use
        the full ``_SoftmaxLoss`` defined above.
    sparse_embeddings: ``bool``, optional (default: False)
        Passed on to ``SampledSoftmaxLoss`` if True.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 contextualizer: Seq2SeqEncoder,
                 dropout: float = None,
                 loss_scale: Union[float, str] = 1.0,
                 num_samples: int = None,
                 sparse_embeddings: bool = False,
                 initializer: InitializerApplicator = None) -> None:
        super().__init__(vocab=vocab,
                         text_field_embedder=text_field_embedder,
                         contextualizer=contextualizer,
                         dropout=dropout,
                         loss_scale=loss_scale,
                         num_samples=num_samples,
                         sparse_embeddings=sparse_embeddings,
                         bidirectional=True,
                         initializer=initializer)
