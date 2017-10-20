from typing import Dict, Optional

from overrides import overrides
import torch
from torch.nn.modules.linear import Linear

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2VecEncoder, TextFieldEmbedder, FeedForward
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import F1Measure


@Model.register("ontoemma")
class OntoEmma(Model):

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 rnn_encoder: Seq2VecEncoder,
                 siamese_feedforward: FeedForward,
                 decision_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(OntoEmma, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.rnn_encoder = rnn_encoder
        self.siamese_feedforward = siamese_feedforward
        self.decision_feedforward = decision_feedforward
        self.sigmoid = torch.nn.Sigmoid()

        if text_field_embedder.get_output_dim() != rnn_encoder.get_input_dim():
            raise ConfigurationError("The output dimension of the text_field_embedder must match the "
                                     "input dimension of the rnn_encoder. Found {} and {}, "
                                     "respectively.".format(text_field_embedder.get_output_dim(),
                                                            rnn_encoder.get_input_dim()))

        if rnn_encoder.get_output_dim() != siamese_feedforward.get_input_dim():
            raise ConfigurationError("The output dimension of the rnn_encoder must match the "
                                     "input dimension of the siamese_feedforward net. Found {} and {}, "
                                     "respectively.".format(rnn_encoder.get_output_dim(),
                                                            siamese_feedforward.get_input_dim()))

        # print("Text field embedder output dim: %i" % text_field_embedder.get_output_dim())
        # print("RNN encoder input dim: %i" % rnn_encoder.get_input_dim())
        # print("RNN encoder output dim: %i" % rnn_encoder.get_output_dim())
        # print("Siamese Feedforward input dim: %i" % siamese_feedforward.get_input_dim())
        # print("Siamese Feedforward output dim: %i" % siamese_feedforward.get_output_dim())
        # print("Decision Feedforward input dim: %i" % decision_feedforward.get_input_dim())
        # print("Decision Feedforward output dim: %i" % decision_feedforward.get_output_dim())

        self.accuracy = F1Measure(positive_label=1)
        self.loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                s_ent_name: Dict[str, torch.LongTensor],
                t_ent_name: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """

        """
        embedded_s_ent_name = self.text_field_embedder(s_ent_name)
        embedded_t_ent_name = self.text_field_embedder(t_ent_name)

        s_ent_name_mask = get_text_field_mask(s_ent_name)
        encoded_s_ent_name = self.rnn_encoder(embedded_s_ent_name, s_ent_name_mask)

        t_ent_name_mask = get_text_field_mask(t_ent_name)
        encoded_t_ent_name = self.rnn_encoder(embedded_t_ent_name, t_ent_name_mask)

        s_ent_name_output = self.siamese_feedforward(encoded_s_ent_name)
        t_ent_name_output = self.siamese_feedforward(encoded_t_ent_name)

        aggregate_input = torch.cat([s_ent_name_output, t_ent_name_output], dim=-1)

        predicted_label = self.sigmoid(self.decision_feedforward(aggregate_input))

        output_dict = {"predicted_label": predicted_label}

        if label is not None:
            loss = self.loss(predicted_label, label.long().view(-1))
            self.accuracy(predicted_label, label.squeeze(-1))
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple position-wise argmax over each token, converts indices to string labels, and
        adds a ``"tags"`` key to the dictionary with the result.
        """
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        p, r, f1 = self.accuracy.get_metric(reset)
        return {
            'precision': p,
            'recall': r,
            'f1': f1
        }

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'OntoEmma':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        rnn_encoder = Seq2VecEncoder.from_params(params.pop("rnn_encoder"))
        siamese_feedforward = FeedForward.from_params(params.pop("siamese_feedforward"))
        decision_feedforward = FeedForward.from_params(params.pop("decision_feedforward"))

        init_params = params.pop('initializer', None)
        reg_params = params.pop('regularizer', None)
        initializer = (InitializerApplicator.from_params(init_params)
                       if init_params is not None
                       else InitializerApplicator())
        regularizer = RegularizerApplicator.from_params(reg_params) if reg_params is not None else None

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   rnn_encoder=rnn_encoder,
                   siamese_feedforward=siamese_feedforward,
                   decision_feedforward=decision_feedforward,
                   initializer=initializer,
                   regularizer=regularizer)
