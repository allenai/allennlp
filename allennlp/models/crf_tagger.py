from typing import Dict, List

import numpy
from overrides import overrides
import torch
from torch.nn.modules.linear import Linear
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder, ConditionalRandomField
from allennlp.models.model import Model
from allennlp.nn.util import get_text_field_mask, viterbi_decode
from allennlp.training.metrics import SpanBasedF1Measure

START_TAG = "@@START@@"
END_TAG = "@@END@@"

@Model.register("crf_tagger")
class CrfTagger(Model):
    """
    The ``CrfTagger`` encodes a sequence of text with a stacked ``Seq2SeqEncoder``,
    then uses a Conditional Random Field model to predict a tag for each token in the sequence.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    stacked_encoder : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and predicting output tags.
    label_namespace: ``str``, optional (default=``"labels"``)
        We'll need to add special START and END tags to whatever namespace holds our labels.
        Unless you did something unusual, the default value should be what you want.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 stacked_encoder: Seq2SeqEncoder,
                 label_namespace: str = "labels") -> None:
        super().__init__(vocab)

        self.text_field_embedder = text_field_embedder

        # Make sure we have START and END tags
        start_tag = vocab.add_token_to_namespace(START_TAG, label_namespace)
        end_tag = vocab.add_token_to_namespace(END_TAG, label_namespace)
        self.num_tags = self.vocab.get_vocab_size(label_namespace)

        self.stacked_encoder = stacked_encoder

        self.tag_projection_layer = TimeDistributed(Linear(self.stacked_encoder.get_output_dim(),
                                                           self.num_tags))

        self.crf = ConditionalRandomField(self.num_tags, start_tag, end_tag)

        self.span_metric = SpanBasedF1Measure(vocab, tag_namespace="labels")

        if text_field_embedder.get_output_dim() != stacked_encoder.get_input_dim():
            raise ConfigurationError("The output dimension of the text_field_embedder must match the "
                                     "input dimension of the phrase_encoder. Found {} and {}, "
                                     "respectively.".format(text_field_embedder.get_output_dim(),
                                                            stacked_encoder.get_input_dim()))

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
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
        tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels of shape
            ``(batch_size, num_tokens)``.

        Returns
        -------
        An output dictionary consisting of:

        logits: torch.FloatTensor
            The logits that are the output of the ``tag_projection_layer``
        mask: torch.ByteTensor
            The text field mask for the input tokens
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised. Only computed if gold label ``tags`` are provided.
        """
        embedded_text_input = self.text_field_embedder(tokens)
        mask = get_text_field_mask(tokens)
        encoded_text = self.stacked_encoder(embedded_text_input, mask)

        logits = self.tag_projection_layer(encoded_text)

        # The CRF layer only produces a ``loss`` output, so we need to include these
        # in case we want to decode the outputs.
        output = {"logits": logits, "mask": mask}

        # Use Viterbi to find most likely tags
        predicted_tags = self._viterbi_tags(output)

        if tags is not None:
            log_likelihood = self.crf.forward(logits, tags, mask)
            output["loss"] = -log_likelihood

            # Represent viterbi tags as "class probabilities" that we can
            # feed into the `span_metric`
            class_probabilities = logits * 0.
            for i, instance_tags in enumerate(predicted_tags):
                for j, tag in enumerate(instance_tags):
                    k = self.vocab.get_token_index(tag, "labels")
                    class_probabilities[i, j, k] = 1

            self.span_metric(class_probabilities, tags, mask)

        return output

    def _viterbi_tags(self, output_dict: Dict[str, torch.Tensor]) -> List[List[str]]:
        """
        Uses viterbi algorithm to find most likely tags
        """
        logits, mask = output_dict["logits"], output_dict["mask"]

        # TODO(joelgrus): Get rid of this after markn makes changes to ``Predictor``
        if isinstance(logits, numpy.ndarray):
            logits, mask = torch.from_numpy(logits), torch.from_numpy(mask)
        elif isinstance(logits, torch.autograd.Variable):
            logits, mask = logits.data, mask.data

        if logits.ndimension() == 3:
            predictions_list = [logits[i] for i in range(logits.shape[0])]
            mask_list = [mask[i] for i in range(mask.shape[0])]
        else:
            predictions_list = [logits]
            mask_list = [mask]
        all_tags = []
        for prediction, mask in zip(predictions_list, mask_list):
            # The CRF transitions are (next_state, prev_state),
            # but ``viterbi_decode`` expects (prev_state, next_state)
            viterbi_path, _ = viterbi_decode(prediction, self.crf.transitions.data.transpose(1, 0))
            tags = [self.vocab.get_token_from_index(ix, "labels") for ix in viterbi_path]
            all_tags.append(tags)
        output_dict["tags"] = all_tags

        return all_tags

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metric_dict = self.span_metric.get_metric(reset=reset)
        return {x: y for x, y in metric_dict.items() if "overall" in x}

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'CrfTagger':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        stacked_encoder = Seq2SeqEncoder.from_params(params.pop("stacked_encoder"))

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   stacked_encoder=stacked_encoder)
