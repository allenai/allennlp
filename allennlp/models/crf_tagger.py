from typing import Dict, List, Optional

import numpy
from overrides import overrides
import torch
from torch.nn.modules.linear import Linear

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder, ConditionalRandomField
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
import allennlp.nn.util as util
from allennlp.training.metrics import SpanBasedF1Measure

START_TAG = "@@START@@"
END_TAG = "@@END@@"

@Model.register("crf_tagger")
class CrfTagger(Model):
    """
    The ``CrfTagger`` encodes a sequence of text with a ``Seq2SeqEncoder``,
    then uses a Conditional Random Field model to predict a tag for each token in the sequence.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    encoder : ``Seq2SeqEncoder``
        The encoder that we will use in between embedding tokens and predicting output tags.
    label_namespace : ``str``, optional (default=``"labels"``)
        We'll need to add special START and END tags to whatever namespace holds our labels.
        Unless you did something unusual, the default value should be what you want.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 label_namespace: str = "labels",
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)
        self.label_namespace = label_namespace
        self.text_field_embedder = text_field_embedder

        # Make sure we have START and END tags
        self.start_tag_index = vocab.add_token_to_namespace(START_TAG, label_namespace)
        self.end_tag_index = vocab.add_token_to_namespace(END_TAG, label_namespace)
        self.num_tags = self.vocab.get_vocab_size(label_namespace)

        self.encoder = encoder
        self.tag_projection_layer = TimeDistributed(Linear(self.encoder.get_output_dim(),
                                                           self.num_tags))
        self.crf = ConditionalRandomField(self.num_tags, self.start_tag_index, self.end_tag_index)

        self.span_metric = SpanBasedF1Measure(vocab, tag_namespace=label_namespace)

        if text_field_embedder.get_output_dim() != encoder.get_input_dim():
            raise ConfigurationError("The output dimension of the text_field_embedder must match the "
                                     "input dimension of the phrase_encoder. Found {} and {}, "
                                     "respectively.".format(text_field_embedder.get_output_dim(),
                                                            encoder.get_input_dim()))
        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                tags: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : ``Dict[str, torch.LongTensor]``, required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        tags : ``torch.LongTensor``, optional (default = ``None``)
            A torch tensor representing the sequence of integer gold class labels of shape
            ``(batch_size, num_tokens)``.

        Returns
        -------
        An output dictionary consisting of:

        logits : ``torch.FloatTensor``
            The logits that are the output of the ``tag_projection_layer``
        mask : ``torch.LongTensor``
            The text field mask for the input tokens
        tags : ``List[List[str]]``
            The predicted tags using the Viterbi algorithm.
        loss : ``torch.FloatTensor``, optional
            A scalar loss to be optimised. Only computed if gold label ``tags`` are provided.
        """
        embedded_text_input = self.text_field_embedder(tokens)
        mask = util.get_text_field_mask(tokens)
        encoded_text = self.encoder(embedded_text_input, mask)

        logits = self.tag_projection_layer(encoded_text)
        predicted_tags = self._viterbi_tags(logits, mask)

        output = {"logits": logits, "mask": mask, "tags": predicted_tags}

        if tags is not None:
            # Add negative log-likelihood as loss
            log_likelihood = self.crf.forward(logits, tags, mask)
            output["loss"] = -log_likelihood

            # Represent viterbi tags as "class probabilities" that we can
            # feed into the `span_metric`
            class_probabilities = logits * 0.
            for i, instance_tags in enumerate(predicted_tags):
                for j, tag in enumerate(instance_tags):
                    k = self.vocab.get_token_index(tag, self.label_namespace)
                    class_probabilities[i, j, k] = 1

            self.span_metric(class_probabilities, tags, mask)

        return output

    def _viterbi_tags(self, logits, mask) -> List[List[str]]:
        """
        Uses viterbi algorithm to find most likely tags
        """
        _, max_seq_length, num_tags = logits.size()

        # The CRF transitions are (next_state, prev_state),
        # but ``viterbi_decode`` expects (prev_state, next_state)
        transitions = self.crf.transitions.data.transpose(1, 0)

        # Get the tensors out of the variables
        logits, mask = logits.data, mask.data

        all_tags = []
        # Pad the max sequence length by 2 to account for start_tag + end_tag.
        tag_sequence = torch.Tensor(max_seq_length + 2, num_tags)

        for prediction, prediction_mask in zip(logits, mask):
            sequence_length = torch.sum(prediction_mask)

            # Start with everything totally unlikely
            tag_sequence.fill_(-10000.)
            # At timestep 0 we must have the START_TAG
            tag_sequence[0, self.start_tag_index] = 0.
            # At steps 1, ..., sequence_length we just use the incoming prediction
            tag_sequence[1:(sequence_length + 1)] = prediction[:sequence_length]
            # And at the last timestep we must have the END_TAG
            tag_sequence[sequence_length + 1, self.end_tag_index] = 0.

            # We pass the tags and the transitions to ``viterbi_decode``.
            viterbi_path, _ = util.viterbi_decode(tag_sequence[:(sequence_length + 2)], transitions)
            # Get rid of START and END sentinels and replace the indices with the actual tags.
            tags = [self.vocab.get_token_from_index(index, self.label_namespace)
                    for index in viterbi_path[1:-1]]
            all_tags.append(tags)

        return all_tags

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metric_dict = self.span_metric.get_metric(reset=reset)
        return {x: y for x, y in metric_dict.items() if "overall" in x}

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'CrfTagger':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        encoder = Seq2SeqEncoder.from_params(params.pop("encoder"))
        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        params.assert_empty(cls.__name__)

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   encoder=encoder,
                   initializer=initializer,
                   regularizer=regularizer)
