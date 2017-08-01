from typing import Any, Dict, Tuple

import torch

from allennlp.common import Params, constants
from allennlp.common.tensor import get_text_field_mask, masked_softmax, last_dim_softmax, weighted_sum
from allennlp.common.tensor import arrays_to_variables
from allennlp.data import Vocabulary
from allennlp.data.fields import TextField
from allennlp.models.model import Model
from allennlp.modules import FeedForward, MatrixAttention
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed, TextFieldEmbedder


@Model.register("decomposable_attention")
class DecomposableAttention(Model):
    """
    This ``Model`` implements the Decomposable Attention model described in "A Decomposable
    Attention Model for Natural Language Inference", by Parikh et al., 2016, with some optional
    enhancements before the decomposable attention actually happens.  Parikh's original model
    allowed for computing an "intra-sentence" attention before doing the decomposable entailment
    step.  We generalize this to any :class:`Seq2SeqEncaoder` that can be applied to the premise
    and/or the hypothesis before computing entailment.

    The basic outline of this model is to get an embedded representation of each word in the
    premise and hypothesis, align words between the two, compare the aligned phrases, and make a
    final entailment decision based on this aggregated comparison.  Each step in this process uses
    a feedforward network to modify the representation.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``premise`` and ``hypothesis`` ``TextFields`` we get as input to the
        model.
    premise_encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        After embedding the premise, we can optionally apply an encoder.  If this is ``None``, we
        will do nothing.
    hypothesis_encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        After embedding the hypothesis, we can optionally apply an encoder.  If this is ``None``,
        we will use the ``premise_encoder`` for the encoding (doing nothing if ``premise_encoder``
        is also ``None``).
    attend_feedforward : ``FeedForward``
        This feedforward network is applied to the encoded sentence representations before the
        similarity matrix is computed between words in the premise and words in the hypothesis.
    similarity_function : ``SimilarityFunction``
        This is the similarity function used when computing the similarity matrix between words in
        the premise and words in the hypothesis.
    compare_feedforward : ``FeedForward``
        This feedforward network is applied to the aligned premise and hypothesis representations,
        individually.
    aggregate_feedforward : ``FeedForward``
        This final feedforward network is applied to the concatenated, summed result of the
        ``compare_feedforward`` network, and its output is used as the entailment class logits.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 premise_encoder: Optional[Seq2SeqEncoder] = None,
                 hypothesis_encoder: Optional[Seq2SeqEncoder] = None,
                 attend_feedforward: FeedForward,
                 similarity_function: SimilarityFunction,
                 compare_feedforward: FeedForward,
                 aggregate_feedforward: FeedForward) -> None:
        super(DecomposableAttention, self).__init__()

        self._vocab = vocab
        self._text_field_embedder = text_field_embedder
        self._premise_encoder = premise_encoder
        self._hypothesis_encoder = hypothesis_encoder or premise_encoder
        self._attend_feedforward = TimeDistributed(attend_feedforward)
        self._matrix_attention = MatrixAttention(similarity_function)
        self._compare_feedforward = TimeDistributed(compare_feedforward)
        self._aggregate_feedforward = aggregate_feedforward
        self._num_labels = vocab.get_vocab_size(namespace="labels")
        if aggregate_feedforward.get_output_dim() != self._num_labels:
            raise ConfigurationError("Final output dimension (%d) must equal num labels (%d)" %
                                     (aggregate_feedforward.get_output_dim(), self._num_labels))

        self._loss = torch.nn.CrossEntropyLoss()

        # TODO(mattg): figure out default initialization here

    def forward(self,  # type: ignore
                premise: Dict[str, torch.LongTensor],
                hypothesis: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        premise : Dict[str, torch.LongTensor]
            From a ``TextField``
        hypothesis : Dict[str, torch.LongTensor]
            From a ``TextField``
        label : torch.IntTensor, optional (default = None)
            From a ``LabelField``

        Returns
        -------
        An output dictionary consisting of:

        label_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing unnormalised log
            probabilities of the entailment label.
        label_probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing probabilities of the
            entailment label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.

        """
        embedded_premise = self._text_field_embedder(premise)
        embedded_hypothesis = self._text_field_embedder(hypothesis)
        batch_size, premise_length, _ = embedded_premise.size()
        hypothesis_length = embedded_hypothesis.size(1)
        premise_mask = get_text_field_mask(premise).float()
        hypothesis_mask = get_text_field_mask(hypothesis).float()

        if self._premise_encoder:
            embedded_premise = self._premise_encoder(embedded_premise)
        if self._hypothesis_encoder:
            embedded_hypothesis = self._hypothesis_encoder(embedded_hypothesis)

        projected_premise = self._attend_feedforward(embedded_premise)
        projected_hypothesis = self._attend_feedforward(embedded_hypothesis)
        # Shape: (batch_size, premise_length, hypothesis_length)
        similarity_matrix = self._matrix_attention(projected_premise, projected_hypothesis)

        # Shape: (batch_size, premise_length, hypothesis_length)
        p2h_attention = last_dim_softmax(similarity_matrix, hypothesis_mask)
        # Shape: (batch_size, premise_length, embedding_dim)
        attended_hypothesis = weighted_sum(embedded_hypothesis, p2h_attention)

        # Shape: (batch_size, hypothesis_length, premise_length)
        h2p_attention = last_dim_softmax(similarity_matrix.transpose(1, 2), premise_mask)
        # Shape: (batch_size, hypothesis_length, embedding_dim)
        attended_premise = weighted_sum(embedded_premise, h2p_attention)

        premise_compare_input = torch.cat([embedded_premise, attended_hypothesis], dim=-1)
        hypothesis_compare_input = torch.cat([embedded_hypothesis, attended_premise], dim=-1)

        compared_premise = self._compare_feedforward(premise_compare_input)
        # TODO(mattg): use broadcasting once pytorch 0.2 is released.
        compared_premise = compared_premise * premise_mask.expand_as(compared_premise)
        # Shape: (batch_size, compare_dim)
        compared_premise = compared_premise.sum(dim=1).squeeze(1)

        compared_hypothesis = self._compare_feedforward(hypothesis_compare_input)
        # TODO(mattg): use broadcasting once pytorch 0.2 is released.
        compared_hypothesis = compared_hypothesis * hypothesis_mask.expand_as(compared_hypothesis)
        # Shape: (batch_size, compare_dim)
        compared_hypothesis = compared_hypothesis.sum(dim=1).squeeze(1)

        aggregate_input = torch.cat([compared_premise, compared_hypothesis], dim=-1)
        label_logits = self._aggregate_feedforward(aggregate_input)
        label_probs = torch.nn.functional.softmax(label_logits)

        output_dict = {"label_logits": label_logits, "label_probs": label_probs}

        if label:
            if label.dim() == 2:
                _, label = label.max(-1)
            loss = self._loss(label_logits:, label.view(-1))
            output_dict["loss"] = loss

        return output_dict

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'BidirectionalAttentionFlow':
        """
        With an empty ``params`` argument, this will instantiate a decomposable attention model
        with the same configuration as published in the original paper, as long as you've set
        ``allennlp.common.constants.GLOVE_PATH`` to the location of your gzipped 300-dimensional
        glove vectors.

        If you want to change parameters, the keys in the ``params`` object must match the
        constructor arguments above.
        """
        default_embedder_params = {
                'tokens': {
                        'type': 'embedding',
                        'projection_dim': 200,
                        'pretrained_file': constants.GLOVE_PATH,
                        'trainable': False
                        }
                }
        embedder_params = params.pop("text_field_embedder", default_embedder_params)
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)

        premise_encoder_params = params.pop("premise_encoder", None)
        if premise_encoder_params is not None:
            premise_encoder = Seq2SeqEncoder.from_params(premise_encoder_params)
        else:
            premise_encoder = None

        hypothesis_encoder_params = params.pop("hypothesis_encoder", None)
        if hypothesis_encoder_params is not None:
            hypothesis_encoder = Seq2SeqEncoder.from_params(hypothesis_encoder_params)
        else:
            hypothesis_encoder = None

        default_attend_params = {
                'input_dim': 200,
                'num_layers': 2,
                'hidden_dims': 200,
                'activations': 'relu'
                }
        attend_params = params.pop('attend_feedforward', default_attend_params)
        attend_feedforward = FeedForward.from_params(attend_params)

        default_similarity_function_params = {'type': 'dot_product'}
        similarity_function_params = params.pop("similarity_function", default_similarity_function_params)
        similarity_function = SimilarityFunction.from_params(similarity_function_params)

        default_compare_params = {
                'input_dim': 400,
                'num_layers': 2,
                'hidden_dims': 200,
                'activations': 'relu'
                }
        compare_params = params.pop('compare_feedforward', default_compare_params)
        compare_feedforward = FeedForward.from_params(compare_params)

        default_aggregate_params = {
                'input_dim': 400,
                'num_layers': 2,
                'hidden_dims': [200, 3],
                'activations': ['relu', 'linear']
                }
        aggregate_params = params.pop('aggregate_feedforward', default_aggregate_params)
        aggregate_feedforward = FeedForward.from_params(aggregate_params)

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   premise_encoder=premise_encoder,
                   hypothesis_encoder=hypothesis_encoder,
                   attend_feedforward=attend_feedforward,
                   similarity_function=similarity_function,
                   compare_feedforward=compare_feedforward,
                   aggregate_feedforward=aggregate_feedforward)
