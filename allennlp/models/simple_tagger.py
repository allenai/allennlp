from typing import List

import torch
from torch.nn.modules.linear import Linear
from torch.nn.modules import LSTM
import torch.nn.functional as F

from ..training.model import Model
from ..layers.embeddings import Embedding
from ..data.vocabulary import Vocabulary
from ..data.fields.text_field import TextField


class SimpleTagger(Model):
    """
    This ``SimpleTagger`` simply encodes a sequence of text with some number of stacked
    ``seq2seq_encoders``, then predicts a tag at each index.
    """

    def __init__(self,
                 vocabulary: Vocabulary,
                 embedding_dim: int = 100,
                 hidden_size: int = 200):
        super(SimpleTagger, self).__init__()

        self.vocabulary = vocabulary
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        self.embedding = Embedding(self.embedding_dim,
                                   self.vocabulary.get_vocab_size("tokens"))

        self.stacked_encoders = LSTM(self.embedding_dim,
                                     self.hidden_size,
                                     self.num_layers,
                                     batch_first=True)

        self.tag_projection_layer = Linear(self.hidden_size,
                                           self.vocabulary.get_vocab_size("tags"))
        self.sequence_loss = torch.nn.CrossEntropyLoss()

    def forward(self,
                text_input: torch.IntTensor,
                sequence_tags: torch.IntTensor = None):
        """
        Parameters
        ----------
        text_input :
        sequence_tags : torch.IntTensor, optional (default = None)
            A torch tensor representing the sequence of gold labels.
            These can either be integer indexes or one hot arrays of
            labels, so of shape (batch_size, sequence_length) or of
            shape (batch_size, sequence_length, vocabulary_size).

        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape (batch_size, sequence_length, tag_vocab_size)
            representing unnormalised log probabilities of the tag classes.
        loss: : torch.FloatTensor, optional
            A scalar loss to be optimised.

        """
        embedded_text_input = self.embedding(text_input)
        encoded_text = self.stacked_encoders(embedded_text_input)
        logits = self.tag_projection_layer(encoded_text)
        class_probabilities = F.softmax(logits)
        output_dict = {"logits": logits, "class_probabilities": class_probabilities}

        if sequence_tags:
            # Averaged over sequence length and batch.
            reshaped_log_probs = logits.view(-1, self.vocabulary.get_vocab_size("tags"))

            # NLL criterion takes integer labels, not one hot.
            if sequence_tags.dim() == 3:
                _, sequence_tags = sequence_tags.max(-1)
            loss = self.sequence_loss(reshaped_log_probs, sequence_tags.view(-1))
            output_dict["loss"] = loss

        return output_dict

    def tag(self, text: List[str]):

        text_field = TextField(tokens=text)
        padding_lengths = text_field.get_padding_lengths()
        text_field.pad(padding_lengths)
        sentence_arrays = text_field.index(self.vocabulary)
        output_dict = self.forward(text_input=sentence_arrays)
        predictions = output_dict["class_probabilities"]
        _, indices = predictions.max(-1).numpy().astype("int32")

        return [self.vocabulary.get_token_from_index(x, namespace="tags") for x in indices]
