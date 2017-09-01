from typing import Dict, Optional

import torch

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Instance, Vocabulary
from allennlp.data.fields import TextField
from allennlp.models.model import Model
from allennlp.modules import FeedForward, MatrixAttention
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed, TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask, last_dim_softmax, weighted_sum
from allennlp.nn.util import arrays_to_variables
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn.initializers import InitializerApplicator


@Model.register("decomposable_attention")
class DecomposableAttention(Model):
    """
    This ``Model`` implements the Decomposable Attention model described in `"A Decomposable
    Attention Model for Natural Language Inference"
    <https://www.semanticscholar.org/paper/A-Decomposable-Attention-Model-for-Natural-Languag-Parikh-T%C3%A4ckstr%C3%B6m/07a9478e87a8304fc3267fa16e83e9f3bbd98b27>`_
    by Parikh et al., 2016, with some optional enhancements before the decomposable attention
    actually happens.  Parikh's original model allowed for computing an "intra-sentence" attention
    before doing the decomposable entailment step.  We generalize this to any
    :class:`Seq2SeqEncoder` that can be applied to the premise and/or the hypothesis before
    computing entailment.

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
    initializer : ``InitializerApplicator``
        We will use this to initialize the parameters in the model, calling ``initializer(self)``.
    premise_encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        After embedding the premise, we can optionally apply an encoder.  If this is ``None``, we
        will do nothing.
    hypothesis_encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        After embedding the hypothesis, we can optionally apply an encoder.  If this is ``None``,
        we will use the ``premise_encoder`` for the encoding (doing nothing if ``premise_encoder``
        is also ``None``).
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 attend_feedforward: FeedForward,
                 similarity_function: SimilarityFunction,
                 compare_feedforward: FeedForward,
                 aggregate_feedforward: FeedForward,
                 initializer: InitializerApplicator,
                 premise_encoder: Optional[Seq2SeqEncoder] = None,
                 hypothesis_encoder: Optional[Seq2SeqEncoder] = None) -> None:
        super(DecomposableAttention, self).__init__(vocab)

        self._text_field_embedder = text_field_embedder
        self._attend_feedforward = TimeDistributed(attend_feedforward)
        self._matrix_attention = MatrixAttention(similarity_function)
        self._compare_feedforward = TimeDistributed(compare_feedforward)
        self._aggregate_feedforward = aggregate_feedforward
        self._premise_encoder = premise_encoder
        self._hypothesis_encoder = hypothesis_encoder or premise_encoder

        self._num_labels = vocab.get_vocab_size(namespace="labels")
        if aggregate_feedforward.get_output_dim() != self._num_labels:
            raise ConfigurationError("Final output dimension (%d) must equal num labels (%d)" %
                                     (aggregate_feedforward.get_output_dim(), self._num_labels))

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)

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
        premise_mask = get_text_field_mask(premise).float()
        hypothesis_mask = get_text_field_mask(hypothesis).float()

        if self._premise_encoder:
            embedded_premise = self._premise_encoder(embedded_premise, premise_mask)
        if self._hypothesis_encoder:
            embedded_hypothesis = self._hypothesis_encoder(embedded_hypothesis, hypothesis_mask)

        projected_premise = self._attend_feedforward(embedded_premise)
        projected_hypothesis = self._attend_feedforward(embedded_hypothesis)
        # Shape: (batch_size, premise_length, hypothesis_length)
        similarity_matrix = self._matrix_attention(projected_premise, projected_hypothesis)

        # Shape: (batch_size, premise_length, hypothesis_length)
        p2h_attention = last_dim_softmax(similarity_matrix, hypothesis_mask)
        # Shape: (batch_size, premise_length, embedding_dim)
        attended_hypothesis = weighted_sum(embedded_hypothesis, p2h_attention)

        # Shape: (batch_size, hypothesis_length, premise_length)
        h2p_attention = last_dim_softmax(similarity_matrix.transpose(1, 2).contiguous(), premise_mask)
        # Shape: (batch_size, hypothesis_length, embedding_dim)
        attended_premise = weighted_sum(embedded_premise, h2p_attention)

        premise_compare_input = torch.cat([embedded_premise, attended_hypothesis], dim=-1)
        hypothesis_compare_input = torch.cat([embedded_hypothesis, attended_premise], dim=-1)

        compared_premise = self._compare_feedforward(premise_compare_input)
        compared_premise = compared_premise * premise_mask.unsqueeze(-1)
        # Shape: (batch_size, compare_dim)
        compared_premise = compared_premise.sum(dim=1)

        compared_hypothesis = self._compare_feedforward(hypothesis_compare_input)
        compared_hypothesis = compared_hypothesis * hypothesis_mask.unsqueeze(-1)
        # Shape: (batch_size, compare_dim)
        compared_hypothesis = compared_hypothesis.sum(dim=1)

        aggregate_input = torch.cat([compared_premise, compared_hypothesis], dim=-1)
        label_logits = self._aggregate_feedforward(aggregate_input)
        label_probs = torch.nn.functional.softmax(label_logits)

        output_dict = {"label_logits": label_logits, "label_probs": label_probs}

        if label is not None:
            loss = self._loss(label_logits, label.long().view(-1))
            self._accuracy(label_logits, label.squeeze(-1))
            output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
                'accuracy': self._accuracy.get_metric(reset),
                }

    def predict_entailment(self, premise: TextField, hypothesis: TextField) -> Dict[str, torch.Tensor]:
        """
        Given a premise and a hypothesis sentence, predict the entailment relationship between
        them.  Note that in the paper, a null token was appended to each sentence, to allow for
        words to align to nothing in the other sentence.  If you've trained your model with a null
        token, you probably want to include it here, too.

        Parameters
        ----------
        premise : ``TextField``
        hypothesis : ``TextField``

        Returns
        -------
        A Dict containing:

        label_probs : torch.FloatTensor
            A tensor of shape ``(num_labels,)`` representing probabilities of the entailment label.
        """
        instance = Instance({"premise": premise, "hypothesis": hypothesis})
        instance.index_fields(self.vocab)
        model_input = arrays_to_variables(instance.as_array_dict(),
                                          add_batch_dimension=True,
                                          for_training=False)
        output_dict = self.forward(**model_input)

        # Remove batch dimension, as we only had one input.
        label_probs = output_dict["label_probs"].data.squeeze(0)
        return {'label_probs': label_probs.numpy()}

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'DecomposableAttention':
        embedder_params = params.pop("text_field_embedder")
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

        attend_feedforward = FeedForward.from_params(params.pop('attend_feedforward'))
        similarity_function = SimilarityFunction.from_params(params.pop("similarity_function"))
        compare_feedforward = FeedForward.from_params(params.pop('compare_feedforward'))
        aggregate_feedforward = FeedForward.from_params(params.pop('aggregate_feedforward'))
        initializer = InitializerApplicator.from_params(params.pop("initializer", []))

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   attend_feedforward=attend_feedforward,
                   similarity_function=similarity_function,
                   compare_feedforward=compare_feedforward,
                   aggregate_feedforward=aggregate_feedforward,
                   initializer=initializer,
                   premise_encoder=premise_encoder,
                   hypothesis_encoder=hypothesis_encoder)
