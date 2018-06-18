from typing import Dict, Optional

from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules.biaffine_attention import BiaffineAttention
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, get_range_vector, get_device_of, last_dim_log_softmax


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
    use_mst_decoding_for_validation : ``bool``, optional (default = True).
        Whether to use Edmond's algorithm to find the optimal MST during validation.
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

        check_dimensions_match(text_field_embedder.get_output_dim(), encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")

        self.use_mst_decoding_for_validation = use_mst_decoding_for_validation

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                words: Dict[str, torch.LongTensor],
                pos_tags: torch.LongTensor,
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
        # TODO remove need for squeeze here.
        attended_arcs = self.arc_attention(head_arc_representation,
                                           child_arc_representation,
                                           mask.float(),
                                           mask.float()).squeeze(1)

        batch_size, timesteps, _ = attended_arcs.size()
        # shape (batch_size,)
        range_vector = get_range_vector(batch_size, get_device_of(attended_arcs))

        minus_inf = -1e8
        minus_mask = (1 - float_mask) * minus_inf
        attended_arcs = attended_arcs + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)
        # shape (batch_size, timesteps, timesteps)
        normalised_arc_logits = last_dim_log_softmax(attended_arcs,
                                                     mask) * float_mask.unsqueeze(2) * float_mask.unsqueeze(1)

        # TODO this is wrong (needs zeroed out diag + mask)
        if head_indices is None:
            _, head_indices = normalised_arc_logits.max(-1)

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
        # We don't care about predictions for the ROOT token, so we remove it from the loss.
        arc_loss = arc_loss[1:, :]
        type_loss = type_loss[1:, :]

        # The number of valid positions is equal to the number of unmasked elements minus
        # 1 per sequence in the batch, to account for the symbolic HEAD token.
        valid_positions = mask.sum() - batch_size

        arc_nll = -arc_loss.sum() / valid_positions.float()
        type_nll = -type_loss.sum() / valid_positions.float()

        output_dict = {
                "arc_loss": arc_nll,
                "type_loss": type_nll,
                "loss": arc_nll + type_nll,
                "arc_logits": normalised_arc_logits,
                "head_type_logits": normalised_head_type_logits,
                "mask": mask
                }

        return output_dict

    def _get_head_types(self,
                        head_type_representation: torch.Tensor,
                        child_type_representation: torch.Tensor,
                        head_indices: torch.Tensor):
        batch_size = head_type_representation.size(0)
        # shape (batch_size,)
        range_vector = get_range_vector(batch_size, get_device_of(head_type_representation))

        # shape (batch_size, timesteps, type_representation_dim)
        selected_head_type_representations = head_type_representation[range_vector,
                                                                      head_indices.data.t()].transpose(0, 1)
        selected_head_type_representations = selected_head_type_representations.contiguous()
        # shape (batch_size, timesteps, num_head_tags)
        head_type_logits = self.type_bilinear(selected_head_type_representations,
                                              child_type_representation)
        return head_type_logits

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'DependencyParser':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        encoder = Seq2SeqEncoder.from_params(params.pop("encoder"))

        type_representation_dim = params.pop_int("type_representation_dim")
        arc_representation_dim = params.pop_int("arc_representation_dim")
        use_mst_decoding_for_validation = params.pop("use_mst_decoding_for_validation", True)
        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))
        params.assert_empty(cls.__name__)
        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   type_representation_dim=type_representation_dim,
                   arc_representation_dim=arc_representation_dim,
                   use_mst_decoding_for_validation=use_mst_decoding_for_validation,
                   encoder=encoder,
                   initializer=initializer,
                   regularizer=regularizer)
