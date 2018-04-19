from typing import Dict, List

from overrides import overrides
import torch

from allennlp.common import Params
from allennlp.modules import Attention, FeedForward
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.nn import util
from allennlp.training.metrics import F1Measure
from allennlp.data.fields.production_rule_field import ProductionRuleArray
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import TextFieldEmbedder
from allennlp.models.model import Model


@Model.register("agenda-predictor")
class AgendaPredictor(Model):
    """
    ``AgendaPredictor`` is a model for generating agendas conditioned on sentences, which can be
    used to guide a semantic parser that learns to search for logical forms.
    This model is essentially a probabilistic version of
    ``allennlp.semparse.worlds.nlvr_world.NlvrWorld.get_agenda_for_sentence``.

    Parameters
    ----------
    vocab : ``Vocabulary``
        Passed to super-class.
    sentence_embedder : ``TextFieldEmbedder``
        Embedder for inputs.
    action_embedding_dim : ``int``
        Dimensionality of action embeddings.
    attention_function : ``SimilarityFunction``
        Attention function to use for attending to the sentence given an action embedding.
    output_projector : ``FeedForward``
        Feed-forward network that projects the concatenation of attended sentence and action
        embedding. We apply a final softmax layer on top of this to project it to a dimensionality
        of 2, corresponding to whether the given action should be in agenda or not.
    rule_namespace : ``str``, optional (default=rule_labels)
        Vocabulary namespace for production rules.
    """
    # TODO(pradeep): Move this class to allennlp.models.semantic_parsing
    def __init__(self,
                 vocab: Vocabulary,
                 sentence_embedder: TextFieldEmbedder,
                 action_embedding_dim: int,
                 attention_function: SimilarityFunction,
                 output_projector: FeedForward,
                 rule_namespace: str = 'rule_labels') -> None:
        super(AgendaPredictor, self).__init__(vocab)
        self._sentence_embedder = sentence_embedder
        self._action_embedding_dim = action_embedding_dim
        self._num_actions = vocab.get_vocab_size(rule_namespace)
        initial_action_embedding = torch.FloatTensor(self._num_actions, self._action_embedding_dim)
        # We don't really need an embedding module for this because we will not call forward on it.
        # So directly defining a Parameter.
        self._action_embedding = torch.nn.Parameter(initial_action_embedding)
        torch.nn.init.xavier_uniform(self._action_embedding)  # Using Glorot Uniform initialization
        self._sentence_embedding_dim = sentence_embedder.get_output_dim()
        assert output_projector.input_dim == self._sentence_embedding_dim + self._action_embedding_dim
        self._output_projector = output_projector
        self._final_projection = torch.nn.Linear(self._output_projector.get_output_dim(), 2)
        self._sentence_attention = Attention(attention_function)
        self._loss_function = torch.nn.CrossEntropyLoss()
        self._f1_metric = F1Measure(self.vocab.get_token_index("in", namespace="labels"))

    def forward(self,  # type: ignore
                sentence: Dict[str, torch.LongTensor],
                all_actions: List[List[ProductionRuleArray]],
                target_actions: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        # Logic for taking an indexed sentence, and predicting what actions the target logical form
        # would contain. We embed the sentence, and for each action, attend over the embedded
        # sentence, and use a feed-forward network that projects the concatenation of the attended
        # sentence and the action embedding to predict a single value, the probability that the
        # action is "in". The loss is a cross-entropy between the target actions and predicted
        # actions.
        # (batch_size, sentence_length, sentence_embedding_dim)
        embedded_sentence = self._sentence_embedder(sentence)
        batch_size, _, _ = embedded_sentence.size()
        # (batch_size, sentence_length)
        sentence_mask = util.get_text_field_mask(sentence).float()
        expanded_action_embedding = self._action_embedding.unsqueeze(0).expand(batch_size,
                                                                               self._num_actions,
                                                                               self._action_embedding_dim)
        actions_attended_sentence = []
        for action_index in range(len(all_actions[0])):
            # (batch_size, action_embedding_dim)
            expanded_action = expanded_action_embedding[:, action_index, :]
            # (batch_size, sentence_length)
            sentence_attention = self._sentence_attention(expanded_action,
                                                          embedded_sentence,
                                                          sentence_mask)
            # (batch_size, sentence_embedding_dim)
            action_attended_sentence = util.weighted_sum(embedded_sentence, sentence_attention)
            actions_attended_sentence.append(action_attended_sentence)
        # (num_actions, batch_size, sentence_embedding_dim)
        attended_sentence = torch.stack(actions_attended_sentence)
        attended_sentence = attended_sentence.view(batch_size,
                                                   self._num_actions,
                                                   self._sentence_embedding_dim)
        # (batch_size, num_actions, sentence_embedding_dim + action_embedding_dim)
        projection_input = torch.cat([attended_sentence, expanded_action_embedding], -1)
        # (batch_size, num_actions, projection_output_dim)
        projection_output = self._output_projector(projection_input)
        # (batch_size, num_actions, 2)
        predicted_actions = torch.nn.functional.softmax(self._final_projection(projection_output), dim=-1)
        # TODO(pradeep): Add F1 metric.
        predicted_action_indices = []
        for prediction in predicted_actions:
            action_indices = []
            for index, prob in enumerate(prediction.data.cpu()):
                if prob[self.vocab.get_token_index("in", namespace="labels")] > 0.5:
                    action_indices.append(index)
            predicted_action_indices.append(action_indices)
        outputs = {"predicted_actions": predicted_action_indices}
        if target_actions is not None:
            # (batch_size, num_actions)
            target_actions = target_actions.squeeze(2)
            # We are computing a separate cross entropy loss for each action that could be in the
            # agenda. That is, we're looking at as many binary classifications as there are actions.
            # Pytorch does not like it when we directly pass a 3D target. So we're looping over
            # actions here and averaging the losses.
            losses = []
            for i in range(self._num_actions):
                losses.append(self._loss_function(predicted_actions[:, i], target_actions[:, i]))
            outputs["loss"] = torch.mean(torch.cat(losses))
            self._f1_metric(predicted_actions, target_actions)
        return outputs

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        precision, recall, f1_measure = self._f1_metric.get_metric(reset)
        return {
                "precision": precision,
                "recall": recall,
                "f1_measure": f1_measure
        }

    @classmethod
    def from_params(cls, vocab, params: Params) -> 'AgendaPredictor':
        sentence_embedder = TextFieldEmbedder.from_params(vocab, params.pop("sentence_embedder"))
        action_embedding_dim = params.pop_int("action_embedding_dim")
        attention_function = SimilarityFunction.from_params(params.pop("attention_function"))
        output_projector = FeedForward.from_params(params.pop("output_projector"))
        rule_namespace = params.pop("rule_namespace", "rule_labels")
        params.assert_empty(cls.__name__)
        return cls(vocab,
                   sentence_embedder=sentence_embedder,
                   action_embedding_dim=action_embedding_dim,
                   attention_function=attention_function,
                   output_projector=output_projector,
                   rule_namespace=rule_namespace)
