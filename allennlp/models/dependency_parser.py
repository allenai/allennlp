from typing import Dict, Optional, Tuple

from overrides import overrides
import torch
import torch.nn.functional as F
import numpy

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, Embedding
from allennlp.modules.biaffine_attention import BiaffineAttention
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, get_range_vector
from allennlp.nn.util import get_device_of, last_dim_log_softmax, get_lengths_from_binary_sequence_mask
from allennlp.nn.decoding.chu_liu_edmonds import decode_mst
from allennlp.training.metrics import AttachmentScores


@Model.register("dependency_parser")
class DependencyParser(Model):
    """
    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    encoder : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use to generate representations
        of tokens.
    type_representation_dim : ``int``, required.
        The dimension of the MLPs used for head type prediction.
    arc_representation_dim : ``int``, required.
        The dimension of the MLPs used for head arc prediction.
    pos_tag_embedding : ``Embedding``, optional.
        Used to embed the ``pos_tags`` ``SequenceLabelField`` we get as input to the model.
    use_mst_decoding_for_validation : ``bool``, optional (default = True).
        Whether to use Edmond's algorithm to find the optimal minimum spanning tree during validation.
        If false, decoding is greedy.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 type_representation_dim: int,
                 arc_representation_dim: int,
                 pos_tag_embedding: Embedding = None,
                 use_mst_decoding_for_validation: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(DependencyParser, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.encoder = encoder

        encoder_dim = encoder.get_output_dim()
        self.head_arc_projection = torch.nn.Linear(encoder_dim, arc_representation_dim)
        self.child_arc_projection = torch.nn.Linear(encoder_dim, arc_representation_dim)
        self.arc_attention = BiaffineAttention(arc_representation_dim, arc_representation_dim, 1)

        num_labels = self.vocab.get_vocab_size("head_tags")
        self.head_type_projection = torch.nn.Linear(encoder_dim, type_representation_dim)
        self.child_type_projection = torch.nn.Linear(encoder_dim, type_representation_dim)
        self.type_bilinear = torch.nn.modules.Bilinear(type_representation_dim,
                                                       type_representation_dim,
                                                       num_labels)

        self._pos_tag_embedding = pos_tag_embedding or None
        representation_dim = text_field_embedder.get_output_dim()
        if pos_tag_embedding is not None:
            representation_dim += pos_tag_embedding.get_output_dim()
        check_dimensions_match(representation_dim, encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")

        self.use_mst_decoding_for_validation = use_mst_decoding_for_validation

        self._attachement_scores = AttachmentScores()
        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                words: Dict[str, torch.LongTensor],
                pos_tags: torch.LongTensor = None,
                head_tags: torch.LongTensor = None,
                head_indices: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels of shape
            ``(batch_size, num_tokens)``.

        Returns
        -------
        An output dictionary consisting of:
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        embedded_text_input = self.text_field_embedder(words)
        if pos_tags is not None and self._pos_tag_embedding is not None:
            embedded_pos_tags = self._pos_tag_embedding(pos_tags)
            embedded_text_input = torch.cat([embedded_text_input, embedded_pos_tags], -1)
        elif self._pos_tag_embedding is not None:
            raise ConfigurationError("Model uses a POS embedding, but no POS tags were passed.")

        mask = get_text_field_mask(words)
        float_mask = mask.float()
        encoded_text = self.encoder(embedded_text_input, mask)

        # shape (batch_size, timesteps, arc_representation_dim)
        head_arc_representation = F.elu(self.head_arc_projection(encoded_text))
        child_arc_representation = F.elu(self.child_arc_projection(encoded_text))

        # shape (batch_size, timesteps, type_representation_dim)
        head_type_representation = F.elu(self.head_type_projection(encoded_text))
        child_type_representation = F.elu(self.child_type_projection(encoded_text))
        head_type_representation = head_type_representation.contiguous()
        child_type_representation = child_type_representation.contiguous()
        # shape (batch_size, timesteps, timesteps)
        attended_arcs = self.arc_attention(head_arc_representation,
                                           child_arc_representation,
                                           mask.float(),
                                           mask.float()).squeeze(1)

        minus_inf = -1e8
        minus_mask = (1 - float_mask) * minus_inf
        attended_arcs = attended_arcs + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        if self.training or (self.eval and not self.use_mst_decoding_for_validation):
            heads, head_types = self._greedy_decode(head_type_representation,
                                                    child_type_representation,
                                                    attended_arcs,
                                                    mask)
        elif self.use_mst_decoding_for_validation:
            heads, head_types = self._mst_decode(head_type_representation,
                                                 child_type_representation,
                                                 attended_arcs,
                                                 mask)


        if head_indices is not None and head_tags is not None:
            arc_nll, type_nll = self._construct_loss(head_type_representation,
                                                     child_type_representation,
                                                     attended_arcs,
                                                     head_indices,
                                                     head_tags,
                                                     mask)
            loss = arc_nll + type_nll

            # We calculate attatchment scores for the whole sentence
            # but excluding the symbolic ROOT token at the start,
            # which is why we start from the second element in the sequence.
            self._attachement_scores(heads[:, 1:],
                                     head_types[:, 1:],
                                     head_indices[:, 1:],
                                     head_tags[:, 1:],
                                     mask[:, 1:])
        else:
            arc_nll = None
            type_nll = None
            loss = None

        output_dict = {
                "heads": heads,
                "head_types": head_types,
                "arc_loss": arc_nll,
                "type_loss": type_nll,
                "loss": loss,
                "mask": mask
                }

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        head_types = output_dict["head_types"].cpu().detach().numpy()
        heads = output_dict["heads"].cpu().detach().numpy()
        lengths = get_lengths_from_binary_sequence_mask(output_dict["mask"])
        head_type_labels = []
        head_indices = []
        for batch, batch_type, length in zip(heads, head_types, lengths):

            batch = list(batch[1: length])
            batch_type = batch_type[1: length]
            labels = [self.vocab.get_token_from_index(label, "head_tags")
                      for label in batch_type]
            head_type_labels.append(labels)
            head_indices.append(batch)

        output_dict["head_type_labels"] = head_type_labels
        output_dict["head_indices"] = head_indices
        return output_dict

    def _construct_loss(self,
                        head_type_representation: torch.Tensor,
                        child_type_representation: torch.Tensor,
                        attended_arcs: torch.Tensor,
                        head_indices: torch.Tensor,
                        head_tags: torch.Tensor,
                        mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the arc and type loss for a sequence given gold head indices and tags.

        Parameters
        ----------
        head_type_representation : ``torch.Tensor``, required.
            A tensor of shape (batch_size, timesteps, type_representation_dim),
            which will be used to generate predictions for the label type
            for given arcs.
        child_type_representation : ``torch.Tensor``, required
            A tensor of shape (batch_size, timesteps, type_representation_dim),
            which will be used to generate predictions for the label type
            for given arcs.
        attended_arcs : ``torch.Tensor``, required.
            A tensor of shape (batch_size, timesteps, timesteps) used to generate
            a distribution over attachements of a given word to all other words.
        head_indices : ``torch.Tensor``, required.
            A tensor of shape (batch_size, timesteps).
            The indices of the heads for every word.
        head_tags : ``torch.Tensor``, required.
            A tensor of shape (batch_size, timesteps).
            The dependency labels of the heads for every word.
        mask : ``torch.Tensor``, required.
            A mask of shape (batch_size, sequence_length), denoting unpadded
            elements in the sequence.

        Returns
        -------
        arc_nll : ``torch.Tensor``, required.
            The negative log likelihood from the arc loss.
        type_nll : ``torch.Tensor``, required.
            The negative log likelihood from the arc type loss.
        """
        float_mask = mask.float()
        batch_size, timesteps, _ = attended_arcs.size()
        range_vector = get_range_vector(batch_size, get_device_of(attended_arcs))
        # shape (batch_size, timesteps, timesteps)
        normalised_arc_logits = last_dim_log_softmax(attended_arcs,
                                                     mask) * float_mask.unsqueeze(2) * float_mask.unsqueeze(1)

        # shape (batch_size, timesteps, num_head_tags)
        head_type_logits = self._get_head_types(head_type_representation, child_type_representation, head_indices)
        normalised_head_type_logits = last_dim_log_softmax(head_type_logits,
                                                           mask.unsqueeze(-1)) * float_mask.unsqueeze(-1)
        # index matrix with shape (timesteps, batch)
        child_index = torch.arange(0, timesteps).view(timesteps, 1).expand(timesteps, batch_size)
        child_index = child_index.type_as(attended_arcs).long()
        # shape (timesteps, batch_size)
        arc_loss = normalised_arc_logits[range_vector, child_index, head_indices.data.t()]
        type_loss = normalised_head_type_logits[range_vector, child_index, head_tags.data.t()]
        # We don't care about predictions for the symbolic ROOT token's head,
        # so we remove it from the loss.
        arc_loss = arc_loss[1:, :]
        type_loss = type_loss[1:, :]

        # The number of valid positions is equal to the number of unmasked elements minus
        # 1 per sequence in the batch, to account for the symbolic HEAD token.
        valid_positions = mask.sum() - batch_size

        arc_nll = -arc_loss.sum() / valid_positions.float()
        type_nll = -type_loss.sum() / valid_positions.float()

        return arc_nll, type_nll

    def _greedy_decode(self,
                       head_type_representation: torch.Tensor,
                       child_type_representation: torch.Tensor,
                       attended_arcs: torch.Tensor,
                       mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes the head and head type predictions by decoding the unlabeled arcs
        independently for each word and then again, predicting the head types of
        these greedily chosen arcs indpendently. Note that this method of decoding
        is not guaranteed to produce trees (i.e. there maybe be multiple roots,
        or cycles when children are attached to their parents).

        Parameters
        ----------
        head_type_representation : ``torch.Tensor``, required.
            A tensor of shape (batch_size, timesteps, type_representation_dim),
            which will be used to generate predictions for the label type
            for given arcs.
        child_type_representation : ``torch.Tensor``, required
            A tensor of shape (batch_size, timesteps, type_representation_dim),
            which will be used to generate predictions for the label type
            for given arcs.
        attended_arcs : ``torch.Tensor``, required.
            A tensor of shape (batch_size, timesteps, timesteps) used to generate
            a distribution over attachements of a given word to all other words.

        Returns
        -------
        heads : ``torch.Tensor``
            A tensor of shape (batch_size, timesteps) representing the
            greedily decoded heads of each word.
        head_types : ``torch.Tensor``
            A tensor of shape (batch_size, timesteps) representing the
            types of the greedily decoded heads of each word.
        """
        # Mask the diagonal, because the head of a word can't be itself.
        attended_arcs = attended_arcs + torch.diag(attended_arcs.new(mask.size(1)).fill_(-numpy.inf))
        # Mask padded tokens, because we only want to consider actual words as heads.
        if mask is not None:
            minus_mask = (1 - mask).byte().unsqueeze(2)
            attended_arcs.masked_fill_(minus_mask, -numpy.inf)

        # Compute the heads greedily.
        # shape (batch_size, timesteps)
        _, heads = attended_arcs.max(dim=2)

        # Given the greedily predicted heads, decode their types.
        # shape (batch_size, timesteps, num_head_tags)
        head_type_logits = self._get_head_types(head_type_representation,
                                                child_type_representation,
                                                heads)
        _, head_types = head_type_logits.max(dim=2)
        return heads, head_types

    def _mst_decode(self,
                    head_type_representation: torch.Tensor,
                    child_type_representation: torch.Tensor,
                    attended_arcs: torch.Tensor,
                    mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes the head and head type predictions using the Edmonds' Algorithm
        for finding minimum spanning trees on directed graphs. Nodes in the
        graph are the words in the sentence, and between each pair of nodes,
        there is an edge in each direction, where the weight of the edge corresponds
        to the most likely dependency label probability for that arc. The MST is
        then generated from this directed graph.

        Parameters
        ----------
        head_type_representation : ``torch.Tensor``, required.
            A tensor of shape (batch_size, timesteps, type_representation_dim),
            which will be used to generate predictions for the label type
            for given arcs.
        child_type_representation : ``torch.Tensor``, required
            A tensor of shape (batch_size, timesteps, type_representation_dim),
            which will be used to generate predictions for the label type
            for given arcs.
        attended_arcs : ``torch.Tensor``, required.
            A tensor of shape (batch_size, timesteps, timesteps) used to generate
            a distribution over attachements of a given word to all other words.

        Returns
        -------
        heads : ``torch.Tensor``
            A tensor of shape (batch_size, timesteps) representing the
            greedily decoded heads of each word.
        head_types : ``torch.Tensor``
            A tensor of shape (batch_size, timesteps) representing the
            types of the greedily decoded heads of each word.
        """
        batch_size, timesteps, type_representation_dim = head_type_representation.size()

        lengths = mask.data.sum(dim=1).long().cpu().numpy()

        expanded_shape = [batch_size, timesteps, timesteps, type_representation_dim]
        head_type_representation = head_type_representation.unsqueeze(2)
        head_type_representation = head_type_representation.expand(*expanded_shape).contiguous()
        child_type_representation = child_type_representation.unsqueeze(1)
        child_type_representation = child_type_representation.expand(*expanded_shape).contiguous()
        # Shape (batch_size, timesteps, timesteps, num_head_tags)
        pairwise_head_logits = self.type_bilinear(head_type_representation, child_type_representation)

        # Shape (batch, length, length, num_labels)
        normalized_pairwise_head_logits = F.log_softmax(pairwise_head_logits, dim=3).permute(0, 3, 1, 2)

        # Mask padded tokens, because we only want to consider actual words as heads.
        minus_inf = -1e8
        minus_mask = (1 - mask.float()) * minus_inf
        attended_arcs = attended_arcs + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        # Shape (batch_size, timesteps, timesteps)
        normalized_arc_logits = F.log_softmax(attended_arcs, dim=2)
        # Shape (batch_size, num_head_tags, timesteps, timesteps)
        batch_energy = torch.exp(normalized_arc_logits.unsqueeze(1) + normalized_pairwise_head_logits)

        heads = []
        head_types = []
        for energy, length in zip(batch_energy.detach().cpu().numpy(), lengths):
            head, head_type = decode_mst(energy, length)
            heads.append(head)
            head_types.append(head_type)
        return torch.from_numpy(numpy.stack(heads)), torch.from_numpy(numpy.stack(head_types))


    def _get_head_types(self,
                        head_type_representation: torch.Tensor,
                        child_type_representation: torch.Tensor,
                        head_indices: torch.Tensor) -> torch.Tensor:
        """
        Decodes the head types given the head and child type representations
        and a tensor of head indices to compute types for. Note that these are
        either gold or predicted heads, depending on whether this function is
        being called to compute the loss, or if it's being called during inference.
        Parameters
        ----------
        head_type_representation : ``torch.Tensor``, required.
            A tensor of shape (batch_size, timesteps, type_representation_dim),
            which will be used to generate predictions for the label type
            for given arcs.
        child_type_representation : ``torch.Tensor``, required
            A tensor of shape (batch_size, timesteps, type_representation_dim),
            which will be used to generate predictions for the label type
            for given arcs.
        head_indices : ``torch.Tensor``, required.
            A tensor of shape (batch_size, timesteps). The indices of the heads
            for every word.

        Returns
        -------
        head_type_logits : ``torch.Tensor``
            A tensor of shape (batch_size, timesteps, num_head_tags),
            representing logits for predicting a distribution over tags
            for each arc.
        """
        batch_size = head_type_representation.size(0)
        # shape (batch_size,)
        range_vector = get_range_vector(batch_size, get_device_of(head_type_representation))

        # TODO(Mark): This can probably replace the code in the "batch_index_select" function in utils.

        # This next statement is quite a complex piece of indexing. What we are doing here is:

        # 1. Indexing in to the batch dimension with an index vector.
        #    Imagine that this momentarily creates a batch_size number of
        #    (timesteps, embedding_dim) shaped tensors.
        # 2. Indexing into the timesteps dimension with the releveant indexes which we
        #    require for this batch element. Because we already created a 'virtual'
        #    (timesteps, embedding_dim) shaped tensor _for this batch element_, we
        #    need to transpose the indices to match the first dimension's shape.
        # 3. This results in a tensor of shape (timesteps, batch_size, embedding_dim),
        #    so we need to transpose back to put the batch dimension first.

        # shape (batch_size, timesteps, type_representation_dim)
        selected_head_type_representations = head_type_representation[range_vector,
                                                                      head_indices.data.t()].transpose(0, 1)
        selected_head_type_representations = selected_head_type_representations.contiguous()
        # shape (batch_size, timesteps, num_head_tags)
        head_type_logits = self.type_bilinear(selected_head_type_representations,
                                              child_type_representation)
        return head_type_logits

    @overrides
    def get_metrics(self, reset: bool = True) -> Dict[str, float]:
        return self._attachement_scores.get_metric(reset)
