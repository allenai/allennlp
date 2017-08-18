from typing import Dict, Any, List, TextIO

import torch
from torch.nn.modules.linear import Linear
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.constants import GLOVE_PATH
from allennlp.common.checks import ConfigurationError
from allennlp.nn.initializers import InitializerApplicator
from allennlp.data import Instance, Vocabulary
from allennlp.data.fields import IndexField, TextField
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn.util import arrays_to_variables, viterbi_decode, get_lengths_from_binary_sequence_mask
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import SpanBasedF1Measure


@Model.register("srl")
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
    initializer : ``InitializerApplicator``
        We will use this to initialize the parameters in the model, calling ``initializer(self)``.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 stacked_encoder: Seq2SeqEncoder,
                 initializer: InitializerApplicator) -> None:
        super(SemanticRoleLabeler, self).__init__()

        self.vocab = vocab
        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("tags")

        # For the span based evaluation, we don't want to consider labels
        # for verb, because the verb index is provided to the model.
        self.span_metric = SpanBasedF1Measure(vocab, tag_namespace="tags", ignore_classes=["V"])

        self.stacked_encoder = stacked_encoder
        self.tag_projection_layer = TimeDistributed(Linear(self.stacked_encoder.get_output_dim(),
                                                           self.num_classes))
        initializer(self)

        if text_field_embedder.get_output_dim() + 1 != stacked_encoder.get_input_dim():
            raise ConfigurationError("The SRL Model uses a binary verb indicator feature, meaning "
                                     "the input dimension of the stacked_encoder must be equal to "
                                     "the output dimension of the text_field_embedder + 1.")

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
        mask = get_text_field_mask(tokens)
        expanded_verb_indicator = verb_indicator.unsqueeze(-1).float()
        # Concatenate the verb feature onto the embedded text. This now
        # has shape (batch_size, sequence_length, embedding_dim + 1).
        embedded_text_with_verb_indicator = torch.cat([embedded_text_input, expanded_verb_indicator], -1)
        batch_size, sequence_length, embedding_dim_with_binary_feature = embedded_text_with_verb_indicator.size()

        if self.stacked_encoder.get_input_dim() != embedding_dim_with_binary_feature:
            raise ConfigurationError("The SRL model uses an indicator feature, which makes "
                                     "the embedding dimension one larger than the value "
                                     "specified. Therefore, the 'input_dim' of the stacked_encoder "
                                     "must be equal to total_embedding_dim + 1.")

        batch_sequence_lengths = get_lengths_from_binary_sequence_mask(mask)
        encoded_text = self.stacked_encoder(embedded_text_with_verb_indicator, batch_sequence_lengths)

        logits = self.tag_projection_layer(encoded_text)
        reshaped_log_probs = logits.view(-1, self.num_classes)
        class_probabilities = F.softmax(reshaped_log_probs).view([batch_size, sequence_length, self.num_classes])
        output_dict = {"logits": logits, "class_probabilities": class_probabilities}
        if tags is not None:
            # Negative log likelihood criterion takes integer labels, not one hot.
            if tags.dim() == 3:
                _, tags = tags.max(-1)
            loss = sequence_cross_entropy_with_logits(logits, tags, mask)
            self.span_metric(class_probabilities, tags, mask)
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
        model_input = arrays_to_variables(instance.as_array_dict(),
                                          add_batch_dimension=True,
                                          for_training=False)
        output_dict = self.forward(**model_input)

        # Remove batch dimension, as we only had one input.
        predictions = output_dict["class_probabilities"].data.squeeze(0)
        transition_matrix = self.get_viterbi_pairwise_potentials()

        max_likelihood_sequence, _ = viterbi_decode(predictions, transition_matrix)
        tags = [self.vocab.get_token_from_index(x, namespace="tags")
                for x in max_likelihood_sequence]

        return {"tags": tags, "class_probabilities": predictions.numpy()}

    def get_metrics(self, reset: bool = False):
        metric_dict = self.span_metric.get_metric(reset=reset)
        if self.training:
            # This can be a lot of metrics, as there are 3 per class.
            # During training, we only really care about the overall
            # metrics, so we filter for them here.
            # TODO(Mark): This is fragile and should be replaced with some verbosity level in Trainer.
            return {x: y for x, y in metric_dict if "overall" in x}

        return metric_dict

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
        """
        With an empty ``params`` argument, this will instantiate a SRL model with the same
        configuration as published in the "Deep Semantic Role Labeling - What works and what's
        next" paper, as long as you've set ``allennlp.common.constants.GLOVE_PATH`` to the
        location of your gzipped 100-dimensional glove vectors.

        If you want to change parameters, the keys in the ``params`` object must match the
        constructor arguments above.
        """
        default_embedder_params = {
                'tokens': {
                        'type': 'embedding',
                        'pretrained_file': GLOVE_PATH,
                        'trainable': True
                        }
                }

        embedder_params = params.pop("text_field_embedder", default_embedder_params)
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)

        default_lstm_params = {
                'type': 'alternating_lstm',
                'input_size': 101,  # Because of the verb_indicator feature.
                'hidden_size': 300,
                'num_layers': 8,
                'recurrent_dropout_probability': 0.1,
                'use_highway': True
                }
        encoder_params = params.pop("stacked_encoder", default_lstm_params)
        stacked_encoder = Seq2SeqEncoder.from_params(encoder_params)

        default_initializer_params = {'bias': {'type': 'normal', 'std': 0.1},
                                      'default': 'orthogonal'}

        initializer_params = params.pop('initializer', default_initializer_params)
        initializer = InitializerApplicator.from_params(initializer_params)

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   stacked_encoder=stacked_encoder,
                   initializer=initializer)

def write_to_conll_eval_file(prediction_file: TextIO,
                             gold_file: TextIO,
                             verb_index: int,
                             sentence: List[str],
                             prediction: List[str],
                             gold_labels: List[str]):
    """
    Prints predicate argument predictions and gold labels for a single verbal
    predicate in a sentence to two provided file references.

    Parameters
    ----------
    prediction_file : TextIO, required.
        A file reference to print predictions to.
    gold_file : TextIO, required.
        A file reference to print gold labels to.
    verb_index : int, required.
        The index of the verbal predicate in the sentence which
        the gold labels are the arguments for.
    sentence : List[str], required.
        The word tokens.
    prediction : List[str], required.
        The predicted BIO labels.
    gold_labels : List[str], required.
        The gold BIO labels.
    """
    verb_only_sentence = ["-"] * len(sentence)
    verb_only_sentence[verb_index] = sentence[verb_index]

    conll_format_predictions = convert_bio_tags_to_conll_format(prediction)
    conll_format_gold_labels = convert_bio_tags_to_conll_format(gold_labels)

    for word, predicted, gold in zip(verb_only_sentence,
                                     conll_format_predictions,
                                     conll_format_gold_labels):
        prediction_file.write(word.ljust(15))
        prediction_file.write(predicted.rjust(15) + "\n")
        gold_file.write(word.ljust(15))
        gold_file.write(gold.rjust(15) + "\n")
    prediction_file.write("\n")
    gold_file.write("\n")


def convert_bio_tags_to_conll_format(labels: List[str]):
    """
    Converts BIO formatted SRL tags to the format required for evaluation with the
    official CONLL 2005 perl script. Spans are represented by bracketed labels,
    with the labels of words inside spans being the same as those outside spans.
    Beginning spans always have a opening bracket and a closing asterisk (e.g. "(ARG-1*" )
    and closing spans always have a closing bracket (e.g. "*)" ). This applies even for
    length 1 spans, (e.g "(ARG-0*)").

    A full example of the conversion performed:

    [B-ARG-1, I-ARG-1, I-ARG-1, I-ARG-1, I-ARG-1, O]
    [ "(ARG-1*", "*", "*", "*", "*)", "*"]

    Parameters
    ----------
    labels : List[str], required.
        A list of BIO tags to convert to the CONLL span based format.

    Returns
    -------
    A list of labels in the CONLL span based format.
    """
    sentence_length = len(labels)
    conll_labels = []
    for i, label in enumerate(labels):
        if label == "O":
            conll_labels.append("*")
            continue
        new_label = "*"
        # Are we at the beginning of a new span, at the first word in the sentence,
        # or is the label different from the previous one? If so, we are seeing a new label.
        if label[0] == "B" or i == 0 or label[1:] != labels[i - 1][1:]:
            new_label = "(" + label[2:] + new_label
        # Are we at the end of the sentence, is the next word a new span, or is the next
        # word not in a span? If so, we need to close the label span.
        if i == sentence_length - 1 or labels[i + 1][0] == "B" or label[1:] != labels[i + 1][1:]:
            new_label = new_label + ")"
        conll_labels.append(new_label)
    return conll_labels
