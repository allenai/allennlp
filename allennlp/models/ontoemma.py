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
from allennlp.training.metrics import BooleanF1


@Model.register("ontoemma")
class OntoEmma(Model):

    def __init__(self, vocab: Vocabulary,
                 name_text_field_embedder: TextFieldEmbedder,
                 # alias_text_field_embedder: TextFieldEmbedder,
                 name_rnn_encoder: Seq2VecEncoder,
                 # alias_rnn_encoder: Seq2VecEncoder,
                 siamese_feedforward: FeedForward,
                 decision_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(OntoEmma, self).__init__(vocab, regularizer)

        self.name_text_field_embedder = name_text_field_embedder
        # self.alias_text_field_embedder = alias_text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.name_rnn_encoder = name_rnn_encoder
        # self.alias_rnn_encoder = alias_rnn_encoder
        self.siamese_feedforward = siamese_feedforward
        self.decision_feedforward = decision_feedforward
        self.sigmoid = torch.nn.Sigmoid()

        if name_text_field_embedder.get_output_dim() != name_rnn_encoder.get_input_dim():
            raise ConfigurationError("The output dimension of the name_text_field_embedder must match the "
                                     "input dimension of the name_rnn_encoder. Found {} and {}, "
                                     "respectively.".format(name_text_field_embedder.get_output_dim(),
                                                            name_rnn_encoder.get_input_dim()))

        # if alias_text_field_embedder.get_output_dim() != alias_rnn_encoder.get_input_dim():
        #     raise ConfigurationError("The output dimension of the alias_text_field_embedder must match the "
        #                              "input dimension of the alias_rnn_encoder. Found {} and {}, "
        #                              "respectively.".format(alias_text_field_embedder.get_output_dim(),
        #                                                     alias_rnn_encoder.get_input_dim()))

        # if name_rnn_encoder.get_output_dim() + alias_rnn_encoder.get_output_dim() != siamese_feedforward.get_input_dim():
        #     raise ConfigurationError("The output dimension of the two rnn_encoders must match the "
        #                              "input dimension of the siamese_feedforward net. Found {} and {}, "
        #                              "respectively.".format(name_rnn_encoder.get_output_dim() + alias_rnn_encoder.get_output_dim(),
        #                                                     siamese_feedforward.get_input_dim()))

        # print("Text field embedder output dim: %i" % text_field_embedder.get_output_dim())
        # print("Name RNN encoder input dim: %i" % name_rnn_encoder.get_input_dim())
        # print("Name RNN encoder output dim: %i" % name_rnn_encoder.get_output_dim())
        # print("Alias RNN encoder input dim: %i" % alias_rnn_encoder.get_input_dim())
        # print("Alias RNN encoder output dim: %i" % alias_rnn_encoder.get_output_dim())
        # print("Siamese Feedforward input dim: %i" % siamese_feedforward.get_input_dim())
        # print("Siamese Feedforward output dim: %i" % siamese_feedforward.get_output_dim())
        # print("Decision Feedforward input dim: %i" % decision_feedforward.get_input_dim())
        # print("Decision Feedforward output dim: %i" % decision_feedforward.get_output_dim())

        self.accuracy = BooleanF1()
        self.loss = torch.nn.BCELoss()

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                s_ent_name: Dict[str, torch.LongTensor],
                t_ent_name: Dict[str, torch.LongTensor],
                # s_ent_aliases: Dict[str, torch.LongTensor],
                # t_ent_aliases: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """

        """
        embedded_s_ent_name = self.name_text_field_embedder(s_ent_name)
        s_ent_name_mask = get_text_field_mask(s_ent_name)
        encoded_s_ent_name = self.name_rnn_encoder(embedded_s_ent_name, s_ent_name_mask)

        embedded_t_ent_name = self.name_text_field_embedder(t_ent_name)
        t_ent_name_mask = get_text_field_mask(t_ent_name)
        encoded_t_ent_name = self.name_rnn_encoder(embedded_t_ent_name, t_ent_name_mask)

        # embedded_s_ent_aliases = self.alias_text_field_embedder(s_ent_aliases)
        # s_ent_alias_mask = get_text_field_mask(s_ent_aliases)
        # encoded_s_ent_alias = self.alias_rnn_encoder(embedded_s_ent_aliases, s_ent_alias_mask)

        # embedded_t_ent_aliases = self.alias_text_field_embedder(t_ent_aliases)
        # t_ent_alias_mask = get_text_field_mask(t_ent_aliases)
        # encoded_t_ent_alias = self.alias_rnn_encoder(embedded_t_ent_aliases, t_ent_alias_mask)

        s_ent_input = encoded_s_ent_name
        t_ent_input = encoded_t_ent_name

        # s_ent_input = torch.cat([encoded_s_ent_name, encoded_s_ent_alias], dim=-1)
        # t_ent_input = torch.cat([encoded_t_ent_name, encoded_t_ent_alias], dim=-1)

        s_ent_output = self.siamese_feedforward(s_ent_input)
        t_ent_output = self.siamese_feedforward(t_ent_input)

        aggregate_input = torch.cat([s_ent_output, t_ent_output], dim=-1)

        decision_output = self.decision_feedforward(aggregate_input)
        sigmoid_output = self.sigmoid(decision_output)

        predicted_label = sigmoid_output.round()
        # print(torch.cat([predicted_label, label.float().squeeze(-1)], dim=-1))

        output_dict = {"predicted_label": predicted_label}

        if label is not None:
            loss = self.loss(sigmoid_output, label.float().view(-1))
            self.accuracy(predicted_label, label)
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
        precision, recall, accuracy, f1 = self.accuracy.get_metric(reset)
        return {
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'f1_score': f1
        }

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'OntoEmma':
        name_text_field_embedder = TextFieldEmbedder.from_params(vocab, params.pop("name_text_field_embedder"))
        # alias_text_field_embedder = TextFieldEmbedder.from_params(vocab, params.pop("alias_text_field_embedder"))
        name_rnn_encoder = Seq2VecEncoder.from_params(params.pop("name_rnn_encoder"))
        # alias_rnn_encoder = Seq2VecEncoder.from_params(params.pop("alias_rnn_encoder"))
        siamese_feedforward = FeedForward.from_params(params.pop("siamese_feedforward"))
        decision_feedforward = FeedForward.from_params(params.pop("decision_feedforward"))

        init_params = params.pop('initializer', None)
        reg_params = params.pop('regularizer', None)
        initializer = (InitializerApplicator.from_params(init_params)
                       if init_params is not None
                       else InitializerApplicator())
        regularizer = RegularizerApplicator.from_params(reg_params) if reg_params is not None else None

        return cls(vocab=vocab,
                   name_text_field_embedder=name_text_field_embedder,
                   # alias_text_field_embedder=alias_text_field_embedder,
                   name_rnn_encoder=name_rnn_encoder,
                   # alias_rnn_encoder=alias_rnn_encoder,
                   siamese_feedforward=siamese_feedforward,
                   decision_feedforward=decision_feedforward,
                   initializer=initializer,
                   regularizer=regularizer)
