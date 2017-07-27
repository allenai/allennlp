from typing import Dict, Any

import torch
from torch.nn.modules.linear import Linear
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.data.fields.text_field import TextField
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.training import Model
from allennlp.experiments.registry import Registry


@Registry.register_model("simple_tagger")
class SimpleTagger(Model):
    """
    This ``SimpleTagger`` simply encodes a sequence of text with a stacked ``Seq2SeqEncoder``, then
    predicts a tag for each token in the sequence.

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
        super(SimpleTagger, self).__init__()

        self.vocab = vocab
        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("tags")
        self.stacked_encoder = stacked_encoder
        self.tag_projection_layer = TimeDistributed(Linear(self.stacked_encoder.get_output_dim(),
                                                           self.num_classes))

        # TODO(Mark): support masking once utility functions are merged.
        self.sequence_loss = torch.nn.CrossEntropyLoss()

    # pylint: disable=arguments-differ
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                tags: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
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
            A torch tensor representing the sequence of gold labels.  These can either be integer
            indexes or one hot arrays of labels, so of shape ``(batch_size, num_tokens)`` or of
            shape ``(batch_size, num_tokens, num_tags)``.

        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            unnormalised log probabilities of the tag classes.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.

        """
        embedded_text_input = self.text_field_embedder(tokens)
        batch_size = embedded_text_input.size()[0]
        encoded_text = self.stacked_encoder(embedded_text_input)

        logits = self.tag_projection_layer(encoded_text)
        reshaped_log_probs = logits.view(-1, self.num_classes)
        class_probabilities = F.softmax(reshaped_log_probs).view([batch_size, -1, self.num_classes])

        output_dict = {"logits": logits, "class_probabilities": class_probabilities}

        if tags:
            # Negative log likelihood criterion takes integer labels, not one hot.
            if tags.dim() == 3:
                _, tags = tags.max(-1)
            loss = self.sequence_loss(reshaped_log_probs, tags.view(-1))
            output_dict["loss"] = loss

        return output_dict

    # pylint: enable=arguments-differ

    def tag(self, text_field: TextField) -> Dict[str, Any]:
        """
        Perform inference on a TextField to produce predicted tags and class probabilities
        over the possible tags.

        Parameters
        ----------
        text_field : ``TextField``, required.
            A ``TextField`` containing the text to be tagged.

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
        text_field.index(self.vocab)
        padding_lengths = text_field.get_padding_lengths()
        array_input = text_field.as_array(padding_lengths)
        # TODO(Mark): Generalise how the array is transformed into a variable after settling the data API.
        # Add a batch dimension by unsqueezing, because pytorch
        # doesn't support inputs without one.
        array_input = {"tokens": torch.autograd.Variable(torch.LongTensor(array_input["tokens"])).unsqueeze(0)}
        output_dict = self.forward(tokens=array_input)

        # Remove batch dimension, as we only had one input.
        predictions = output_dict["class_probabilities"].data.squeeze(0)
        _, argmax = predictions.max(-1)
        indices = argmax.squeeze(1).numpy()
        tags = [self.vocab.get_token_from_index(x, namespace="tags") for x in indices]

        return {"tags": tags, "class_probabilities": predictions.numpy()}

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'SimpleTagger':
        text_field_embedder = TextFieldEmbedder.from_params(vocab, params.pop("text_field_embedder"))
        stacked_encoder = Seq2SeqEncoder.from_params(params.pop("stacked_encoder"))
        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   stacked_encoder=stacked_encoder)
