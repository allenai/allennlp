from typing import Dict, Any

import torch
from torch.nn.modules.linear import Linear
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.tensor import arrays_to_variables, viterbi_decode
from allennlp.data import Vocabulary
from allennlp.data.fields import IndexField, TextField
from allennlp.data import Instance
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.models.model import Model


class SemanticRoleLabeler(Model):
    """
    This model performs semantic role labeling using BIO tags using Propbank semantic roles.
    Specifically, it is an implmentation of `Deep Semantic Role Labeling - What works
    and what's next <https://homes.cs.washington.edu/~luheng/files/acl2017_hllz.pdf>`_ .

    This implementation is effectively a series of stacked interleaved LSTMs with highway
    connections, applied to embedded sequences of words concatenated with a binary indicator
    containing whether or not a word is the verbal predicate to generate predictions for in
    the sentence. Additionally, during inference, Viterbi decoding is applied to constrain
    the predictions to contain valid BIO sequences.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    stacked_encoder : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and predicting output tags.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 stacked_encoder: Seq2SeqEncoder) -> None:
        super(SemanticRoleLabeler, self).__init__()

        self.vocab = vocab
        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("tags")

        # NOTE: You must make sure that the "input_dim" of the stacked encoder
        # in your configuration file is equal to self.text_field_embedder.output_dim + 1.
        self.stacked_encoder = stacked_encoder
        self.tag_projection_layer = TimeDistributed(Linear(self.stacked_encoder.get_output_dim(),
                                                           self.num_classes))
        # TODO(Mark): support masking once utility functions are merged.
        self.sequence_loss = torch.nn.CrossEntropyLoss()

    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                verb_indicator: torch.LongTensor,
                tags: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
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
        verb_indicator: torch.LongTensor, required.
            A one-hot/all-zeros ``IndexField`` representation of the position of the verb in the sentence.
            This should have shape (batch_size, num_tokens) and importantly, can be all zeros, in the case
            that the sentence has no verbal predicate.
        tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of gold labels.  These can either be integer
            indexes or one hot arrays of labels, so of shape ``(batch_size, num_tokens)`` or of
            shape ``(batch_size, num_tokens, num_tags)``.

        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            unnormalised log probabilities of the tag classes.
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            a distribution of the tag classes per word.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.

        """
        embedded_text_input = self.text_field_embedder(tokens)
        expanded_verb_indicator = verb_indicator.unsqueeze(-1).float()
        # Concatenate the verb feature onto the embedded text. This now
        # has shape (batch_size, sequence_length, embedding_dim + 1).
        embedded_text_with_verb_indicator = torch.cat([embedded_text_input, expanded_verb_indicator], -1)
        batch_size, sequence_length, _ = embedded_text_with_verb_indicator.size()
        encoded_text = self.stacked_encoder(embedded_text_with_verb_indicator)

        logits = self.tag_projection_layer(encoded_text)
        reshaped_log_probs = logits.view(-1, self.num_classes)
        class_probabilities = F.softmax(reshaped_log_probs).view([batch_size, sequence_length, self.num_classes])
        output_dict = {"logits": logits, "class_probabilities": class_probabilities}
        if tags:
            # Negative log likelihood criterion takes integer labels, not one hot.
            if tags.dim() == 3:
                _, tags = tags.max(-1)
            loss = self.sequence_loss(reshaped_log_probs, tags.view(-1))
            output_dict["loss"] = loss

        return output_dict

    def tag(self, text_field: TextField, verb_indicator: IndexField) -> Dict[str, Any]:
        """
        Perform inference on a ``Instance`` consisting of a single ``TextField`` representing
        the sentence and an ``IndexField`` representing an optional index into the sentence
        denoting a verbal predicate.

        Returned sequence is the maximum likelihood tag sequence under the constraint that
        the sequence must be a valid BIO sequence.

        Parameters
        ----------
        text_field : ``TextField``, required.
            A ``TextField`` containing the text to be tagged.
        verb_indicator: ``IndexField``, required.
            The index of the verb whose arguments we are labeling.

        Returns
        -------
        A Dict containing:

        tags : List[str]
            A list the length of the text input, containing the predicted (argmax) tag
            from the model per token.
        class_probabilities : numpy.Array
            An array of shape (text_input_length, num_classes), where each row is a
            distribution over classes for a given token in the sentence.
        """
        instance = Instance({"tokens": text_field, "verb_indicator": verb_indicator})
        instance.index_fields(self.vocab)
        model_input = instance.as_array(instance.get_padding_lengths())
        torch_input = arrays_to_variables(model_input)
        # TODO(Mark): Make the data API always return tensors with batch dimensions at every abstraction level.
        # Add a batch dimension by unsqueezing, because pytorch doesn't support inputs without one.

        for tensor in torch_input["tokens"].values():
            tensor.data.unsqueeze_(0)
        torch_input["verb_indicator"].data.unsqueeze_(0)

        output_dict = self.forward(**torch_input)

        # Remove batch dimension, as we only had one input.
        predictions = output_dict["class_probabilities"].data.squeeze(0)
        transition_matrix = self.get_viterbi_pairwise_potentials()

        max_likelihood_sequence, _ = viterbi_decode(predictions, transition_matrix)
        tags = [self.vocab.get_token_from_index(x, namespace="tags")
                for x in max_likelihood_sequence]

        return {"tags": tags, "class_probabilities": predictions.numpy()}

    def get_viterbi_pairwise_potentials(self):
        """
        Generate a matrix of pairwise transition potentials for the BIO labels.
        The only constraint implemented here is that I-XXX tags must be preceded
        by either an identical I-XXX tag or a B-XXX tag. In order to achieve this
        constraint, pairs of tags which do not satisfy this constraint have a
        pairwise potential of -inf.

        Returns
        -------
        transition_matrix : torch.Tensor
            A (num_tags, num_tags) matrix of pairwise potentials.
        """
        all_tags = self.vocab.get_index_to_token_vocabulary("tags")
        num_tags = len(all_tags)
        transition_matrix = torch.zeros([num_tags, num_tags])

        for i, previous_tag in all_tags.items():
            for j, tag in all_tags.items():
                # I tags can only be preceded by themselves or
                # their corresponding B tag.
                if i != j and tag[0] == 'I' and not previous_tag == 'B' + tag[1:]:
                    transition_matrix[i, j] = float("-inf")
        return transition_matrix

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'SemanticRoleLabeler':
        text_field_embedder = TextFieldEmbedder.from_params(vocab, params.pop("text_field_embedder"))
        stacked_encoder = Seq2SeqEncoder.from_params(params.pop("stacked_encoder"))
        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   stacked_encoder=stacked_encoder)
