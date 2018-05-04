# TODO: projection dropout with ELMO
#   l2 reg with ELMO
#   multiple ELMO layers
#   doc

from typing import Dict, Optional

import torch
from torch.autograd import Variable

from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, MatrixAttention
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, last_dim_softmax, weighted_sum, replace_masked_values
from allennlp.training.metrics import CategoricalAccuracy

class VariationalDropout(torch.nn.Dropout):
    def forward(self, input):
        """
        input is shape (batch_size, timesteps, embedding_dim)
        Samples one mask of size (batch_size, embedding_dim) and applies it to every time step.
        """
        #ones = Variable(torch.ones(input.shape[0], input.shape[-1]))
        ones = Variable(input.data.new(input.shape[0], input.shape[-1]).fill_(1))
        dropout_mask = torch.nn.functional.dropout(ones, self.p, self.training, inplace=False)
        if self.inplace:
            input *= dropout_mask.unsqueeze(1)
            return None
        else:
            return dropout_mask.unsqueeze(1) * input


@Model.register("esim")
class ESIM(Model):
    """
    This ``Model`` implements the ESIM sequence model described in `"Enhanced LSTM for Natural Language Inference"
    <https://www.semanticscholar.org/paper/Enhanced-LSTM-for-Natural-Language-Inference-Chen-Zhu/83e7654d545fbbaaf2328df365a781fb67b841b4>`_
    by Chen et al., 2017.

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
    premise_encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        After embedding the premise, we can optionally apply an encoder.  If this is ``None``, we
        will do nothing.
    hypothesis_encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        After embedding the hypothesis, we can optionally apply an encoder.  If this is ``None``,
        we will use the ``premise_encoder`` for the encoding (doing nothing if ``premise_encoder``
        is also ``None``).
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 similarity_function: SimilarityFunction,
                 projection_feedforward: FeedForward,
                 inference_encoder: Seq2SeqEncoder,
                 output_feedforward: FeedForward,
                 output_logit: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 dropout: float = 0.5,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._encoder = encoder

        self._matrix_attention = MatrixAttention(similarity_function)
        self._projection_feedforward = projection_feedforward

        self._inference_encoder = inference_encoder

        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
            self.rnn_input_dropout = VariationalDropout(dropout)
        else:
            self.dropout = None
            self.rnn_input_dropout = None

        self._output_feedforward = output_feedforward
        self._output_logit = output_logit

        self._num_labels = vocab.get_vocab_size(namespace="labels")

        check_dimensions_match(text_field_embedder.get_output_dim(), encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")
        check_dimensions_match(encoder.get_output_dim() * 4, projection_feedforward.get_input_dim(),
                               "encoder output dim", "projection feedforward input")
        check_dimensions_match(projection_feedforward.get_output_dim(), inference_encoder.get_input_dim(),
                               "proj feedforward output dim", "inference lstm input dim")

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

        # apply dropout for LSTM
        if self.rnn_input_dropout:
            embedded_premise = self.rnn_input_dropout(embedded_premise)
            embedded_hypothesis = self.rnn_input_dropout(embedded_hypothesis)

        # encode premise and hypothesis
        encoded_premise = self._encoder(embedded_premise, premise_mask)
        encoded_hypothesis = self._encoder(embedded_hypothesis, hypothesis_mask)

        # Shape: (batch_size, premise_length, hypothesis_length)
        similarity_matrix = self._matrix_attention(encoded_premise, encoded_hypothesis)

        # Shape: (batch_size, premise_length, hypothesis_length)
        p2h_attention = last_dim_softmax(similarity_matrix, hypothesis_mask)
        # Shape: (batch_size, premise_length, embedding_dim)
        attended_hypothesis = weighted_sum(encoded_hypothesis, p2h_attention)

        # Shape: (batch_size, hypothesis_length, premise_length)
        h2p_attention = last_dim_softmax(similarity_matrix.transpose(1, 2).contiguous(), premise_mask)
        # Shape: (batch_size, hypothesis_length, embedding_dim)
        attended_premise = weighted_sum(encoded_premise, h2p_attention)

        # the "enhancement" layer
        premise_enhanced = torch.cat(
                [encoded_premise, attended_hypothesis,
                 encoded_premise - attended_hypothesis,
                 encoded_premise * attended_hypothesis],
                dim=-1
        )
        hypothesis_enhanced = torch.cat(
                [encoded_hypothesis, attended_premise, 
                 encoded_hypothesis - attended_premise,
                 encoded_hypothesis * attended_premise],
                dim=-1
        )

        # embedding -> lstm w/ do -> enhanced attention -> dropout_proj, only if ELMO -> ff proj -> lstm w/ do -> dropout -> ff 300 -> dropout -> output

        # add dropout here with ELMO

        # the projection layer down to the model dimension
        # no dropout in projection
        projected_enhanced_premise = self._projection_feedforward(premise_enhanced)
        projected_enhanced_hypothesis = self._projection_feedforward(hypothesis_enhanced)

        # Run the inference layer
        if self.rnn_input_dropout:
            projected_enhanced_premise = self.rnn_input_dropout(projected_enhanced_premise)
            projected_enhanced_hypothesis = self.rnn_input_dropout(projected_enhanced_hypothesis)
        v_ai = self._inference_encoder(projected_enhanced_premise, premise_mask)
        v_bi = self._inference_encoder(projected_enhanced_hypothesis, hypothesis_mask)

        # The pooling layer -- max and avg pooling.
        # (batch_size, model_dim)
        v_a_max, _ = replace_masked_values(
                v_ai, premise_mask.unsqueeze(-1), -1e7
        ).max(dim=1)
        v_b_max, _ = replace_masked_values(
                v_bi, hypothesis_mask.unsqueeze(-1), -1e7
        ).max(dim=1)

        v_a_avg = torch.sum(v_ai * premise_mask.unsqueeze(-1), dim=1) / torch.sum(premise_mask, 1, keepdim=True)
        v_b_avg = torch.sum(v_bi * hypothesis_mask.unsqueeze(-1), dim=1) / torch.sum(hypothesis_mask, 1, keepdim=True)

        # Now concat
        # (batch_size, model_dim * 2 * 4)
        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

        # the final MLP -- apply dropout to input, and MLP applies to output & hidden
        if self.dropout:
            v = self.dropout(v)

        output_hidden = self._output_feedforward(v)
        label_logits = self._output_logit(output_hidden)
        label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

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

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'DecomposableAttention':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)

        encoder = Seq2SeqEncoder.from_params(params.pop("encoder"))
        similarity_function = SimilarityFunction.from_params(params.pop("similarity_function"))
        projection_feedforward = FeedForward.from_params(params.pop('projection_feedforward'))
        inference_encoder = Seq2SeqEncoder.from_params(params.pop("inference_encoder"))
        output_feedforward = FeedForward.from_params(params.pop('output_feedforward'))
        output_logit = FeedForward.from_params(params.pop('output_logit'))
        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        dropout = params.pop("dropout", 0)

        params.assert_empty(cls.__name__)
        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   encoder=encoder,
                   similarity_function=similarity_function,
                   projection_feedforward=projection_feedforward,
                   inference_encoder=inference_encoder,
                   output_feedforward=output_feedforward,
                   output_logit=output_logit,
                   initializer=initializer,
                   dropout=dropout,
                   regularizer=regularizer)
