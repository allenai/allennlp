import logging
from typing import Callable, List, Dict, Tuple, Union

from overrides import overrides

import torch

from allennlp.common import Params
from allennlp.data.fields.production_rule_field import ProductionRuleArray
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.nn.decoding import DecoderTrainer, ExpectedRiskMinimization
from allennlp.nn import util as nn_util
from allennlp.models.model import Model
from allennlp.models.encoder_decoders.nlvr.nlvr_decoder_state import NlvrDecoderState
from allennlp.models.encoder_decoders.nlvr.nlvr_decoder_step import NlvrDecoderStep
from allennlp.models.encoder_decoders.nlvr.nlvr_semantic_parser import NlvrSemanticParser
from allennlp.semparse.worlds import NlvrWorld
from allennlp.training.metrics import Average

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("nlvr_coverage_parser")
class NlvrCoverageSemanticParser(NlvrSemanticParser):
    """
    ``NlvrSemanticCoverageParser`` is an ``NlvrSemanticParser`` that gets around the problem of lack
    of annotated logical forms by maximizing coverage of the output sequences over a prespecified
    agenda. In addition to the signal from coverage, we also compute the denotations given by the
    logical forms and define a hybrid cost based on coverage and denotation errors. The training
    process then minimizes the expected value of this cost over an approximate set of logical forms
    produced by the parser, obtained by performing beam search.

    Parameters
    ----------
    vocab : ``Vocabulary``
        Passed to super-class.
    sentence_embedder : ``TextFieldEmbedder``
        Passed to super-class.
    nonterminal_embedder : ``TextFieldEmbedder``
        Passed to super-class.
    terminal_embedder : ``TextFieldEmbedder``
        Passed to super-class.
    encoder : ``Seq2SeqEncoder``
        Passed to super-class.
    attention_function : ``SimilarityFunction``
        We compute an attention over the input question at each step of the decoder, using the
        decoder hidden state as the query.  This is the similarity function we use for that
        attention.
    beam_size : ``int``
        Beam size for the beam search used during training.
    normalize_beam_score_by_length : ``bool``, optional (default=False)
        Should the log probabilities be normalized by length before renormalizing them? Edunov et
        al. do this in their work, but we found that not doing it works better. It's possible they
        did this because their task is NMT, and longer decoded sequences are not necessarily worse,
        and shouldn't be penalized, while we will mostly want to penalize longer logical forms.
    max_decoding_steps : ``int``
        Maximum number of steps for the beam search during training.
    checklist_cost_weight : ``float``, optional (default=0.8)
        Mixture weight (0-1) for combining coverage cost and denotation cost. As this increases, we
        weigh the coverage cost higher, with a value of 1.0 meaning that we do not care about
        denotation accuracy.
    dynamic_cost_weight : ``Dict[str, Union[int, float]]``, optional (default=None)
        A dict containing keys ``wait_num_epochs`` and ``rate`` indicating the number of steps
        after which we should start decreasing the weight on checklist cost in favor of denotation
        cost, and the rate at which we should do it. We will decrease the weight in the following
        way - ``checklist_cost_weight = checklist_cost_weight - rate * checklist_cost_weight``
        starting at the apropriate epoch.  The weight will remain constant if this is not provided.
    penalize_non_agenda_actions : ``bool``, optional (default=False)
        Should we penalize the model for producing terminal actions that are outside the agenda?
    """
    def __init__(self,
                 vocab: Vocabulary,
                 sentence_embedder: TextFieldEmbedder,
                 nonterminal_embedder: TextFieldEmbedder,
                 terminal_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 attention_function: SimilarityFunction,
                 beam_size: int,
                 max_decoding_steps: int,
                 normalize_beam_score_by_length: bool = False,
                 checklist_cost_weight: float = 0.8,
                 dynamic_cost_weight: Dict[str, Union[int, float]] = None,
                 penalize_non_agenda_actions: bool = False) -> None:
        super(NlvrCoverageSemanticParser, self).__init__(vocab=vocab,
                                                         sentence_embedder=sentence_embedder,
                                                         nonterminal_embedder=nonterminal_embedder,
                                                         terminal_embedder=terminal_embedder,
                                                         encoder=encoder)
        self._agenda_coverage = Average()
        self._decoder_trainer: DecoderTrainer[Callable[[NlvrDecoderState], torch.Tensor]] = \
                ExpectedRiskMinimization(beam_size, normalize_beam_score_by_length, max_decoding_steps)
        action_embedding_dim = nonterminal_embedder.get_output_dim() * 2

        # Instantiating an empty NlvrWorld just to get the number of terminals.
        num_terminals = len(NlvrWorld([]).terminal_productions)
        self._decoder_step = NlvrDecoderStep(encoder_output_dim=self._encoder.get_output_dim(),
                                             action_embedding_dim=action_embedding_dim,
                                             attention_function=attention_function,
                                             checklist_size=num_terminals)
        self._checklist_cost_weight = checklist_cost_weight
        self._dynamic_cost_wait_epochs = None
        self._dynamic_cost_rate = None
        if dynamic_cost_weight:
            self._dynamic_cost_wait_epochs = dynamic_cost_weight["wait_num_epochs"]
            self._dynamic_cost_rate = dynamic_cost_weight["rate"]
        self._penalize_non_agenda_actions = penalize_non_agenda_actions
        self._last_epoch_in_forward: int = None

    @overrides
    def forward(self,  # type: ignore
                sentence: Dict[str, torch.LongTensor],
                worlds: List[List[NlvrWorld]],
                actions: List[List[ProductionRuleArray]],
                agenda: torch.LongTensor,
                labels: torch.LongTensor = None,
                epoch_num: List[int] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Decoder logic for producing type constrained target sequences that maximize coverage of
        their respective agendas, and minimize a denotation based loss.
        """
        # We look at the epoch number and adjust the checklist cost weight if needed here.
        instance_epoch_num = epoch_num[0] if epoch_num is not None else None
        if self._dynamic_cost_rate is not None:
            if instance_epoch_num is None:
                raise RuntimeError("If you want a dynamic cost weight, use the "
                                   "EpochTrackingBucketIterator!")
            if instance_epoch_num != self._last_epoch_in_forward:
                if instance_epoch_num >= self._dynamic_cost_wait_epochs:
                    decrement = self._checklist_cost_weight * self._dynamic_cost_rate
                    self._checklist_cost_weight -= decrement
                    logger.info("Checklist cost weight is now %f", self._checklist_cost_weight)
                self._last_epoch_in_forward = instance_epoch_num
        batch_size = len(worlds)
        action_embeddings, action_indices, initial_action_embedding = self._embed_actions(actions)

        initial_rnn_state = self._get_initial_rnn_state(sentence, initial_action_embedding)
        initial_score_list = [nn_util.new_variable_with_data(agenda, torch.Tensor([0.0])) for i in
                              range(batch_size)]
        label_strings = self._get_label_strings(labels)
        # TODO (pradeep): Assuming all worlds give the same set of valid actions.
        initial_grammar_state = [self._create_grammar_state(worlds[i][0], actions[i]) for i in
                                 range(batch_size)]
        worlds_list = [worlds[i] for i in range(batch_size)]

        # Each instance's agenda is of size (agenda_size, 1)
        agenda_list = [agenda[i] for i in range(batch_size)]
        checklist_targets = []
        all_terminal_actions = []
        checklist_masks = []
        initial_checklist_list = []
        for instance_actions, instance_agenda in zip(actions, agenda_list):
            checklist_info = self._get_checklist_info(instance_agenda, instance_actions)
            checklist_target, terminal_actions, checklist_mask = checklist_info
            checklist_targets.append(checklist_target)
            all_terminal_actions.append(terminal_actions)
            checklist_masks.append(checklist_mask)
            initial_checklist_list.append(nn_util.new_variable_with_size(checklist_target,
                                                                         checklist_target.size(),
                                                                         0))
        initial_state = NlvrDecoderState(batch_indices=list(range(batch_size)),
                                         action_history=[[] for _ in range(batch_size)],
                                         score=initial_score_list,
                                         rnn_state=initial_rnn_state,
                                         grammar_state=initial_grammar_state,
                                         action_embeddings=action_embeddings,
                                         action_indices=action_indices,
                                         possible_actions=actions,
                                         worlds=worlds_list,
                                         label_strings=label_strings,
                                         terminal_actions=all_terminal_actions,
                                         checklist_target=checklist_targets,
                                         checklist_masks=checklist_masks,
                                         checklist=initial_checklist_list)

        agenda_data = [agenda_[:, 0].cpu().data for agenda_ in agenda_list]
        outputs = self._decoder_trainer.decode(initial_state,
                                               self._decoder_step,
                                               self._get_state_cost)
        best_action_sequences = outputs['best_action_sequence']
        self._update_metrics(actions=actions,
                             worlds=worlds,
                             best_action_sequences=best_action_sequences,
                             label_strings=label_strings,
                             agenda_data=agenda_data)
        return outputs

    def _get_checklist_info(self,
                            agenda: torch.LongTensor,
                            all_actions: List[ProductionRuleArray]) -> Tuple[torch.Tensor,
                                                                             torch.Tensor,
                                                                             torch.Tensor]:
        """
        Takes an agenda and a list of all actions and returns a target checklist against which the
        checklist at each state will be compared to compute a loss, indices of ``terminal_actions``,
        and a ``checklist_mask`` that indicates which of the terminal actions are relevant for
        checklist loss computation. If ``self.penalize_non_agenda_actions`` is set to``True``,
        ``checklist_mask`` will be all 1s (i.e., all terminal actions are relevant). If it is set to
        ``False``, indices of all terminals that are not in the agenda will be masked.

        Parameters
        ----------
        ``agenda`` : ``torch.LongTensor``
            Agenda of one instance of size ``(agenda_size, 1)``.
        ``all_actions`` : ``List[ProductionRuleArray]``
            All actions for one instance.
        """
        terminal_indices = []
        target_checklist_list = []
        agenda_indices_set = set([int(x) for x in agenda.squeeze(0).data.cpu().numpy()])
        for index, action in enumerate(all_actions):
            # Each action is a ProductionRuleArray, a dict with keys "left" and "right", and
            # values are tuples where the second element shows whether element is a
            # non_terminal.
            if not action["right"][1]:
                terminal_indices.append([index])
                if index in agenda_indices_set:
                    target_checklist_list.append([1])
                else:
                    target_checklist_list.append([0])
        # We want to return checklist target and terminal actions that are column vectors to make
        # computing softmax over the difference between checklist and target easier.
        # (num_terminals, 1)
        terminal_actions = nn_util.new_variable_with_data(agenda,
                                                          torch.Tensor(terminal_indices))
        # (num_terminals, 1)
        target_checklist = nn_util.new_variable_with_data(agenda,
                                                          torch.Tensor(target_checklist_list))
        if self._penalize_non_agenda_actions:
            # All terminal actions are relevant
            checklist_mask = torch.ones_like(target_checklist)
        else:
            checklist_mask = (target_checklist != 0).float()
        return target_checklist, terminal_actions, checklist_mask

    def _update_metrics(self,
                        actions: List[List[ProductionRuleArray]],
                        worlds: List[List[NlvrWorld]],
                        best_action_sequences: Dict[int, List[int]],
                        label_strings: List[List[str]],
                        agenda_data: List[List[int]]) -> None:
        batch_size = len(worlds)
        for i in range(batch_size):
            batch_actions = actions[i]
            batch_best_sequences = best_action_sequences[i] if i in best_action_sequences else []
            sequence_is_correct = [False]
            in_agenda_ratio = 0.0
            if batch_best_sequences:
                action_strings = [self._get_action_string(batch_actions[rule_id]) for rule_id in
                                  batch_best_sequences]
                terminal_agenda_actions = []
                for rule_id in agenda_data[i]:
                    if rule_id == -1:
                        continue
                    action_string = self._get_action_string(batch_actions[rule_id])
                    right_side = action_string.split(" -> ")[1]
                    if right_side.isdigit() or ('[' not in right_side and len(right_side) > 1):
                        terminal_agenda_actions.append(rule_id)
                actions_in_agenda = [rule_id in batch_best_sequences for rule_id in
                                     terminal_agenda_actions]
                in_agenda_ratio = sum(actions_in_agenda) / len(actions_in_agenda)
                instance_label_strings = label_strings[i]
                instance_worlds = worlds[i]
                sequence_is_correct = self._check_denotation(action_strings,
                                                             instance_label_strings,
                                                             instance_worlds)
            for correct_in_world in sequence_is_correct:
                self._denotation_accuracy(1 if correct_in_world else 0)
            self._consistency(1 if all(sequence_is_correct) else 0)
            self._agenda_coverage(in_agenda_ratio)

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
                'denotation_accuracy': self._denotation_accuracy.get_metric(reset),
                'consistency': self._consistency.get_metric(reset),
                'agenda_coverage': self._agenda_coverage.get_metric(reset)
        }

    def _get_state_cost(self, state: NlvrDecoderState) -> torch.Tensor:
        """
        Return the costs a finished state. Since it is a finished state, the group size will be 1,
        and hence we'll return just one cost.
        """
        if not state.is_finished():
            raise RuntimeError("_get_state_cost() is not defined for unfinished states!")
        instance_checklist_target = state.checklist_target[0]
        instance_checklist = state.checklist[0]
        instance_checklist_mask = state.checklist_mask[0]

        # Our checklist cost is a sum of squared error from where we want to be, making sure we
        # take into account the mask.
        checklist_balance = instance_checklist_target - instance_checklist
        checklist_balance = checklist_balance * instance_checklist_mask
        checklist_cost = torch.sum((checklist_balance) ** 2)

        # This is the number of items on the agenda that we want to see in the decoded sequence.
        # We use this as the denotation cost if the path is incorrect.
        # Note: If we are penalizing the model for producing non-agenda actions, this is not the
        # upper limit on the checklist cost. That would be the number of terminal actions.
        denotation_cost = torch.sum(instance_checklist_target.float())
        checklist_cost = self._checklist_cost_weight * checklist_cost
        # TODO (pradeep): The denotation based cost below is strict. May be define a cost based on
        # how many worlds the logical form is correct in?
        if all(self._check_state_denotations(state)):
            cost = checklist_cost
        else:
            cost = checklist_cost + (1 - self._checklist_cost_weight) * denotation_cost
        return cost

    def _get_state_info(self, state) -> Dict[str, List]:
        """
        This method is here for debugging purposes, in case you want to look at the what the model
        is learning. It may be inefficient to call it while training the model on real data.
        """
        if len(state.batch_indices) == 1 and state.is_finished():
            costs = [float(self._get_state_cost(state).data.cpu().numpy())]
        else:
            costs = []
        model_scores = [float(score.data.cpu().numpy()) for score in state.score]
        all_actions = state.possible_actions[0]
        action_sequences = [[self._get_action_string(all_actions[action]) for action in history]
                            for history in state.action_history]
        agenda_sequences = []
        all_agenda_indices = []
        for agenda, checklist_target in zip(state.terminal_actions, state.checklist_target):
            agenda_indices = []
            for action, is_wanted in zip(agenda, checklist_target):
                action_int = int(action.data.cpu().numpy())
                is_wanted_int = int(is_wanted.data.cpu().numpy())
                if is_wanted_int != 0:
                    agenda_indices.append(action_int)
            agenda_sequences.append([self._get_action_string(all_actions[action])
                                     for action in agenda_indices])
            all_agenda_indices.append(agenda_indices)
        return {"agenda": agenda_sequences,
                "agenda_indices": all_agenda_indices,
                "history": action_sequences,
                "history_indices": state.action_history,
                "costs": costs,
                "scores": model_scores}

    @classmethod
    def from_params(cls, vocab, params: Params) -> 'NlvrCoverageSemanticParser':
        sentence_embedder_params = params.pop("sentence_embedder")
        sentence_embedder = TextFieldEmbedder.from_params(vocab, sentence_embedder_params)
        encoder = Seq2SeqEncoder.from_params(params.pop("encoder"))
        nonterminal_embedder = TextFieldEmbedder.from_params(vocab, params.pop("nonterminal_embedder"))
        terminal_embedder = TextFieldEmbedder.from_params(vocab, params.pop("terminal_embedder"))
        attention_function_type = params.pop("attention_function", None)
        if attention_function_type is not None:
            attention_function = SimilarityFunction.from_params(attention_function_type)
        else:
            attention_function = None
        beam_size = params.pop_int('beam_size')
        normalize_beam_score_by_length = params.pop_bool('normalize_beam_score_by_length', False)
        max_decoding_steps = params.pop_int("max_decoding_steps")
        checklist_cost_weight = params.pop_float("checklist_cost_weight", 0.8)
        dynamic_cost_weight = params.pop("dynamic_cost_weight", None)
        penalize_non_agenda_actions = params.pop_bool("penalize_non_agenda_actions", False)
        params.assert_empty(cls.__name__)
        return cls(vocab,
                   sentence_embedder=sentence_embedder,
                   nonterminal_embedder=nonterminal_embedder,
                   terminal_embedder=terminal_embedder,
                   encoder=encoder,
                   attention_function=attention_function,
                   beam_size=beam_size,
                   max_decoding_steps=max_decoding_steps,
                   normalize_beam_score_by_length=normalize_beam_score_by_length,
                   checklist_cost_weight=checklist_cost_weight,
                   dynamic_cost_weight=dynamic_cost_weight,
                   penalize_non_agenda_actions=penalize_non_agenda_actions)
