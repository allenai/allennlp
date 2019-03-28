from typing import Dict, Optional, Tuple, Any, List
import logging
import copy

from overrides import overrides
import torch
from torch.nn.modules import Dropout
import numpy

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, Embedding, InputVariationalDropout
from allennlp.modules.matrix_attention.bilinear_matrix_attention import BilinearMatrixAttention
from allennlp.modules import FeedForward
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator, Activation
from allennlp.nn.util import get_text_field_mask
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from allennlp.training.metrics import F1Measure

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("graph_parser")
class GraphParser(Model):
    """
    A Parser for arbitrary graph structures.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    encoder : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use to generate representations
        of tokens.
    tag_representation_dim : ``int``, required.
        The dimension of the MLPs used for arc tag prediction.
    arc_representation_dim : ``int``, required.
        The dimension of the MLPs used for arc prediction.
    tag_feedforward : ``FeedForward``, optional, (default = None).
        The feedforward network used to produce tag representations.
        By default, a 1 layer feedforward network with an elu activation is used.
    arc_feedforward : ``FeedForward``, optional, (default = None).
        The feedforward network used to produce arc representations.
        By default, a 1 layer feedforward network with an elu activation is used.
    pos_tag_embedding : ``Embedding``, optional.
        Used to embed the ``pos_tags`` ``SequenceLabelField`` we get as input to the model.
    dropout : ``float``, optional, (default = 0.0)
        The variational dropout applied to the output of the encoder and MLP layers.
    input_dropout : ``float``, optional, (default = 0.0)
        The dropout applied to the embedded text input.
    edge_prediction_threshold : ``int``, optional (default = 0.5)
        The probability at which to consider a scored edge to be 'present'
        in the decoded graph. Must be between 0 and 1.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 tag_representation_dim: int,
                 arc_representation_dim: int,
                 tag_feedforward: FeedForward = None,
                 arc_feedforward: FeedForward = None,
                 pos_tag_embedding: Embedding = None,
                 dropout: float = 0.0,
                 input_dropout: float = 0.0,
                 edge_prediction_threshold: float = 0.5,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(GraphParser, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.encoder = encoder
        self.edge_prediction_threshold = edge_prediction_threshold
        if not 0 < edge_prediction_threshold < 1:
            raise ConfigurationError(f"edge_prediction_threshold must be between "
                                     f"0 and 1 (exclusive) but found {edge_prediction_threshold}.")

        encoder_dim = encoder.get_output_dim()

        self.head_arc_feedforward = arc_feedforward or \
                                        FeedForward(encoder_dim, 1,
                                                    arc_representation_dim,
                                                    Activation.by_name("elu")())
        self.child_arc_feedforward = copy.deepcopy(self.head_arc_feedforward)

        self.arc_attention = BilinearMatrixAttention(arc_representation_dim,
                                                     arc_representation_dim,
                                                     use_input_biases=True)

        num_labels = self.vocab.get_vocab_size("labels")
        self.head_tag_feedforward = tag_feedforward or \
                                        FeedForward(encoder_dim, 1,
                                                    tag_representation_dim,
                                                    Activation.by_name("elu")())
        self.child_tag_feedforward = copy.deepcopy(self.head_tag_feedforward)

        self.tag_bilinear = BilinearMatrixAttention(tag_representation_dim,
                                                    tag_representation_dim,
                                                    label_dim=num_labels)

        self._pos_tag_embedding = pos_tag_embedding or None
        self._dropout = InputVariationalDropout(dropout)
        self._input_dropout = Dropout(input_dropout)

        representation_dim = text_field_embedder.get_output_dim()
        if pos_tag_embedding is not None:
            representation_dim += pos_tag_embedding.get_output_dim()

        check_dimensions_match(representation_dim, encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")
        check_dimensions_match(tag_representation_dim, self.head_tag_feedforward.get_output_dim(),
                               "tag representation dim", "tag feedforward output dim")
        check_dimensions_match(arc_representation_dim, self.head_arc_feedforward.get_output_dim(),
                               "arc representation dim", "arc feedforward output dim")

        self._unlabelled_f1 = F1Measure(positive_label=1)
        self._arc_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self._tag_loss = torch.nn.CrossEntropyLoss(reduction='none')
        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                pos_tags: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None,
                arc_tags: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``.
        pos_tags : torch.LongTensor, optional (default = None)
            The output of a ``SequenceLabelField`` containing POS tags.
        metadata : List[Dict[str, Any]], optional (default = None)
            A dictionary of metadata for each batch element which has keys:
                tokens : ``List[str]``, required.
                    The original string tokens in the sentence.
        arc_tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer indices denoting the parent of every
            word in the dependency parse. Has shape ``(batch_size, sequence_length, sequence_length)``.

        Returns
        -------
        An output dictionary.
        """
        embedded_text_input = self.text_field_embedder(tokens)
        if pos_tags is not None and self._pos_tag_embedding is not None:
            embedded_pos_tags = self._pos_tag_embedding(pos_tags)
            embedded_text_input = torch.cat([embedded_text_input, embedded_pos_tags], -1)
        elif self._pos_tag_embedding is not None:
            raise ConfigurationError("Model uses a POS embedding, but no POS tags were passed.")

        mask = get_text_field_mask(tokens)
        embedded_text_input = self._input_dropout(embedded_text_input)
        encoded_text = self.encoder(embedded_text_input, mask)

        float_mask = mask.float()
        encoded_text = self._dropout(encoded_text)

        # shape (batch_size, sequence_length, arc_representation_dim)
        head_arc_representation = self._dropout(self.head_arc_feedforward(encoded_text))
        child_arc_representation = self._dropout(self.child_arc_feedforward(encoded_text))

        # shape (batch_size, sequence_length, tag_representation_dim)
        head_tag_representation = self._dropout(self.head_tag_feedforward(encoded_text))
        child_tag_representation = self._dropout(self.child_tag_feedforward(encoded_text))
        # shape (batch_size, sequence_length, sequence_length)
        arc_scores = self.arc_attention(head_arc_representation,
                                        child_arc_representation)
        # shape (batch_size, num_tags, sequence_length, sequence_length)
        arc_tag_logits = self.tag_bilinear(head_tag_representation,
                                           child_tag_representation)
        # Switch to (batch_size, sequence_length, sequence_length, num_tags)
        arc_tag_logits = arc_tag_logits.permute(0, 2, 3, 1).contiguous()

        minus_inf = -1e8
        minus_mask = (1 - float_mask) * minus_inf
        arc_scores = arc_scores + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        arc_probs, arc_tag_probs = self._greedy_decode(arc_scores,
                                                       arc_tag_logits,
                                                       mask)

        output_dict = {
                "arc_probs": arc_probs,
                "arc_tag_probs": arc_tag_probs,
                "mask": mask,
                }

        if metadata:
            output_dict["tokens"] = [meta["tokens"] for meta in metadata]

        if arc_tags is not None:
            arc_nll, tag_nll = self._construct_loss(arc_scores=arc_scores,
                                                    arc_tag_logits=arc_tag_logits,
                                                    arc_tags=arc_tags,
                                                    mask=mask)
            output_dict["loss"] = arc_nll + tag_nll
            output_dict["arc_loss"] = arc_nll
            output_dict["tag_loss"] = tag_nll

            # Make the arc tags not have negative values anywhere
            # (by default, no edge is indicated with -1).
            arc_indices = (arc_tags != -1).float()
            tag_mask = float_mask.unsqueeze(1) * float_mask.unsqueeze(2)
            one_minus_arc_probs = 1 - arc_probs
            # We stack scores here because the f1 measure expects a
            # distribution, rather than a single value.
            self._unlabelled_f1(torch.stack([one_minus_arc_probs, arc_probs], -1), arc_indices, tag_mask)

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        arc_tag_probs = output_dict["arc_tag_probs"].cpu().detach().numpy()
        arc_probs = output_dict["arc_probs"].cpu().detach().numpy()
        mask = output_dict["mask"]
        lengths = get_lengths_from_binary_sequence_mask(mask)
        arcs = []
        arc_tags = []
        for instance_arc_probs, instance_arc_tag_probs, length in zip(arc_probs, arc_tag_probs, lengths):

            arc_matrix = instance_arc_probs > self.edge_prediction_threshold
            edges = []
            edge_tags = []
            for i in range(length):
                for j in range(length):
                    if arc_matrix[i, j] == 1:
                        edges.append((i, j))
                        tag = instance_arc_tag_probs[i, j].argmax(-1)
                        edge_tags.append(self.vocab.get_token_from_index(tag, "labels"))
            arcs.append(edges)
            arc_tags.append(edge_tags)

        output_dict["arcs"] = arcs
        output_dict["arc_tags"] = arc_tags
        return output_dict

    def _construct_loss(self,
                        arc_scores: torch.Tensor,
                        arc_tag_logits: torch.Tensor,
                        arc_tags: torch.Tensor,
                        mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the arc and tag loss for an adjacency matrix.

        Parameters
        ----------
        arc_scores : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate a
            binary classification decision for whether an edge is present between two words.
        arc_tag_logits : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length, num_tags) used to generate
            a distribution over edge tags for a given edge.
        arc_tags : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length).
            The labels for every arc.
        mask : ``torch.Tensor``, required.
            A mask of shape (batch_size, sequence_length), denoting unpadded
            elements in the sequence.

        Returns
        -------
        arc_nll : ``torch.Tensor``, required.
            The negative log likelihood from the arc loss.
        tag_nll : ``torch.Tensor``, required.
            The negative log likelihood from the arc tag loss.
        """
        float_mask = mask.float()
        arc_indices = (arc_tags != -1).float()
        # Make the arc tags not have negative values anywhere
        # (by default, no edge is indicated with -1).
        arc_tags = arc_tags * arc_indices
        arc_nll = self._arc_loss(arc_scores, arc_indices) * float_mask.unsqueeze(1) * float_mask.unsqueeze(2)
        # We want the mask for the tags to only include the unmasked words
        # and we only care about the loss with respect to the gold arcs.
        tag_mask = float_mask.unsqueeze(1) * float_mask.unsqueeze(2) * arc_indices

        batch_size, sequence_length, _, num_tags = arc_tag_logits.size()
        original_shape = [batch_size, sequence_length, sequence_length]
        reshaped_logits = arc_tag_logits.view(-1, num_tags)
        reshaped_tags = arc_tags.view(-1)
        tag_nll = self._tag_loss(reshaped_logits, reshaped_tags.long()).view(original_shape) * tag_mask

        valid_positions = tag_mask.sum()

        arc_nll = arc_nll.sum() / valid_positions.float()
        tag_nll = tag_nll.sum() / valid_positions.float()
        return arc_nll, tag_nll

    @staticmethod
    def _greedy_decode(arc_scores: torch.Tensor,
                       arc_tag_logits: torch.Tensor,
                       mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes the head and head tag predictions by decoding the unlabeled arcs
        independently for each word and then again, predicting the head tags of
        these greedily chosen arcs independently.

        Parameters
        ----------
        arc_scores : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachments of a given word to all other words.
        arc_tag_logits : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length, num_tags) used to
            generate a distribution over tags for each arc.
        mask : ``torch.Tensor``, required.
            A mask of shape (batch_size, sequence_length).

        Returns
        -------
        arc_probs : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length, sequence_length) representing the
            probability of an arc being present for this edge.
        arc_tag_probs : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length, sequence_length, sequence_length)
            representing the distribution over edge tags for a given edge.
        """
        # Mask the diagonal, because we don't self edges.
        inf_diagonal_mask = torch.diag(arc_scores.new(mask.size(1)).fill_(-numpy.inf))
        arc_scores = arc_scores + inf_diagonal_mask
        # shape (batch_size, sequence_length, sequence_length, num_tags)
        arc_tag_logits = arc_tag_logits + inf_diagonal_mask.unsqueeze(0).unsqueeze(-1)
        # Mask padded tokens, because we only want to consider actual word -> word edges.
        minus_mask = (1 - mask).byte().unsqueeze(2)
        arc_scores.masked_fill_(minus_mask, -numpy.inf)
        arc_tag_logits.masked_fill_(minus_mask.unsqueeze(-1), -numpy.inf)
        # shape (batch_size, sequence_length, sequence_length)
        arc_probs = arc_scores.sigmoid()
        # shape (batch_size, sequence_length, sequence_length, num_tags)
        arc_tag_probs = torch.nn.functional.softmax(arc_tag_logits, dim=-1)
        return arc_probs, arc_tag_probs

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        precision, recall, f1_measure = self._unlabelled_f1.get_metric(reset)
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1"] = f1_measure
        return metrics
