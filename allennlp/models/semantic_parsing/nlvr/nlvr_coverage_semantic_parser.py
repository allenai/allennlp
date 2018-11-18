import logging
import os
from functools import partial
from typing import Callable, List, Dict, Tuple, Union

from overrides import overrides

import torch

from allennlp.data.fields.production_rule_field import ProductionRule
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.archival import load_archive, Archive
from allennlp.models.model import Model
from allennlp.models.semantic_parsing.nlvr.nlvr_semantic_parser import NlvrSemanticParser
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.nn import Activation
from allennlp.semparse.worlds import NlvrWorld
from allennlp.state_machines.trainers import DecoderTrainer, ExpectedRiskMinimization
from allennlp.state_machines.states import CoverageState, ChecklistStatelet
from allennlp.state_machines.transition_functions import CoverageTransitionFunction
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
    action_embedding_dim : ``int``
        Passed to super-class.
    encoder : ``Seq2SeqEncoder``
        Passed to super-class.
    attention : ``Attention``
        We compute an attention over the input question at each step of the decoder, using the
        decoder hidden state as the query.  Passed to the TransitionFunction.
    beam_size : ``int``
        Beam size for the beam search used during training.
    max_num_finished_states : ``int``, optional (default=None)
        Maximum number of finished states the trainer should compute costs for.
    normalize_beam_score_by_length : ``bool``, optional (default=False)
        Should the log probabilities be normalized by length before renormalizing them? Edunov et
        al. do this in their work, but we found that not doing it works better. It's possible they
        did this because their task is NMT, and longer decoded sequences are not necessarily worse,
        and shouldn't be penalized, while we will mostly want to penalize longer logical forms.
    max_decoding_steps : ``int``
        Maximum number of steps for the beam search during training.
    dropout : ``float``, optional (default=0.0)
        Probability of dropout to apply on encoder outputs, decoder outputs and predicted actions.
    checklist_cost_weight : ``float``, optional (default=0.6)
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
    initial_mml_model_file : ``str`` , optional (default=None)
        If you want to initialize this model using weights from another model trained using MML,
        pass the path to the ``model.tar.gz`` file of that model here.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 sentence_embedder: TextFieldEmbedder,
                 action_embedding_dim: int,
                 encoder: Seq2SeqEncoder,
                 attention: Attention,
                 beam_size: int,
                 max_decoding_steps: int,
                 max_num_finished_states: int = None,
                 dropout: float = 0.0,
                 normalize_beam_score_by_length: bool = False,
                 checklist_cost_weight: float = 0.6,
                 dynamic_cost_weight: Dict[str, Union[int, float]] = None,
                 penalize_non_agenda_actions: bool = False,
                 initial_mml_model_file: str = None) -> None:
        super(NlvrCoverageSemanticParser, self).__init__(vocab=vocab,
                                                         sentence_embedder=sentence_embedder,
                                                         action_embedding_dim=action_embedding_dim,
                                                         encoder=encoder,
                                                         dropout=dropout)
        self._agenda_coverage = Average()
        self._decoder_trainer: DecoderTrainer[Callable[[CoverageState], torch.Tensor]] = \
                ExpectedRiskMinimization(beam_size=beam_size,
                                         normalize_by_length=normalize_beam_score_by_length,
                                         max_decoding_steps=max_decoding_steps,
                                         max_num_finished_states=max_num_finished_states)

        # Instantiating an empty NlvrWorld just to get the number of terminals.
        self._terminal_productions = set(NlvrWorld([]).terminal_productions.values())
        self._decoder_step = CoverageTransitionFunction(encoder_output_dim=self._encoder.get_output_dim(),
                                                        action_embedding_dim=action_embedding_dim,
                                                        input_attention=attention,
                                                        num_start_types=1,
                                                        activation=Activation.by_name('tanh')(),
                                                        predict_start_type_separately=False,
                                                        add_action_bias=False,
                                                        dropout=dropout)
        self._checklist_cost_weight = checklist_cost_weight
        self._dynamic_cost_wait_epochs = None
        self._dynamic_cost_rate = None
        if dynamic_cost_weight:
            self._dynamic_cost_wait_epochs = dynamic_cost_weight["wait_num_epochs"]
            self._dynamic_cost_rate = dynamic_cost_weight["rate"]
        self._penalize_non_agenda_actions = penalize_non_agenda_actions
        self._last_epoch_in_forward: int = None
        # TODO (pradeep): Checking whether file exists here to avoid raising an error when we've
        # copied a trained ERM model from a different machine and the original MML model that was
        # used to initialize it does not exist on the current machine. This may not be the best
        # solution for the problem.
        if initial_mml_model_file is not None:
            if os.path.isfile(initial_mml_model_file):
                archive = load_archive(initial_mml_model_file)
                self._initialize_weights_from_archive(archive)
            else:
                # A model file is passed, but it does not exist. This is expected to happen when
                # you're using a trained ERM model to decode. But it may also happen if the path to
                # the file is really just incorrect. So throwing a warning.
                logger.warning("MML model file for initializing weights is passed, but does not exist."
                               " This is fine if you're just decoding.")

    def _initialize_weights_from_archive(self, archive: Archive) -> None:
        logger.info("Initializing weights from MML model.")
        model_parameters = dict(self.named_parameters())
        archived_parameters = dict(archive.model.named_parameters())
        sentence_embedder_weight = "_sentence_embedder.token_embedder_tokens.weight"
        if sentence_embedder_weight not in archived_parameters or \
           sentence_embedder_weight not in model_parameters:
            raise RuntimeError("When initializing model weights from an MML model, we need "
                               "the sentence embedder to be a TokenEmbedder using namespace called "
                               "tokens.")
        for name, weights in archived_parameters.items():
            if name in model_parameters:
                if name == "_sentence_embedder.token_embedder_tokens.weight":
                    # The shapes of embedding weights will most likely differ between the two models
                    # because the vocabularies will most likely be different. We will get a mapping
                    # of indices from this model's token indices to the archived model's and copy
                    # the tensor accordingly.
                    vocab_index_mapping = self._get_vocab_index_mapping(archive.model.vocab)
                    archived_embedding_weights = weights.data
                    new_weights = model_parameters[name].data.clone()
                    for index, archived_index in vocab_index_mapping:
                        new_weights[index] = archived_embedding_weights[archived_index]
                    logger.info("Copied embeddings of %d out of %d tokens",
                                len(vocab_index_mapping), new_weights.size()[0])
                else:
                    new_weights = weights.data
                logger.info("Copying parameter %s", name)
                model_parameters[name].data.copy_(new_weights)

    def _get_vocab_index_mapping(self, archived_vocab: Vocabulary) -> List[Tuple[int, int]]:
        vocab_index_mapping: List[Tuple[int, int]] = []
        for index in range(self.vocab.get_vocab_size(namespace='tokens')):
            token = self.vocab.get_token_from_index(index=index, namespace='tokens')
            archived_token_index = archived_vocab.get_token_index(token, namespace='tokens')
            # Checking if we got the UNK token index, because we don't want all new token
            # representations initialized to UNK token's representation. We do that by checking if
            # the two tokens are the same. They will not be if the token at the archived index is
            # UNK.
            if archived_vocab.get_token_from_index(archived_token_index, namespace="tokens") == token:
                vocab_index_mapping.append((index, archived_token_index))
        return vocab_index_mapping

    @overrides
    def forward(self,  # type: ignore
                sentence: Dict[str, torch.LongTensor],
                worlds: List[List[NlvrWorld]],
                actions: List[List[ProductionRule]],
                agenda: torch.LongTensor,
                identifier: List[str] = None,
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
            if self.training and instance_epoch_num is None:
                raise RuntimeError("If you want a dynamic cost weight, use the "
                                   "EpochTrackingBucketIterator!")
            if instance_epoch_num != self._last_epoch_in_forward:
                if instance_epoch_num >= self._dynamic_cost_wait_epochs:
                    decrement = self._checklist_cost_weight * self._dynamic_cost_rate
                    self._checklist_cost_weight -= decrement
                    logger.info("Checklist cost weight is now %f", self._checklist_cost_weight)
                self._last_epoch_in_forward = instance_epoch_num
        batch_size = len(worlds)

        initial_rnn_state = self._get_initial_rnn_state(sentence)
        initial_score_list = [next(iter(sentence.values())).new_zeros(1, dtype=torch.float)
                              for i in range(batch_size)]
        # TODO (pradeep): Assuming all worlds give the same set of valid actions.
        initial_grammar_state = [self._create_grammar_state(worlds[i][0], actions[i]) for i in
                                 range(batch_size)]

        label_strings = self._get_label_strings(labels) if labels is not None else None
        # Each instance's agenda is of size (agenda_size, 1)
        # TODO(mattg): It looks like the agenda is only ever used on the CPU.  In that case, it's a
        # waste to copy it to the GPU and then back, and this should probably be a MetadataField.
        agenda_list = [agenda[i] for i in range(batch_size)]
        initial_checklist_states = []
        for instance_actions, instance_agenda in zip(actions, agenda_list):
            checklist_info = self._get_checklist_info(instance_agenda, instance_actions)
            checklist_target, terminal_actions, checklist_mask = checklist_info

            initial_checklist = checklist_target.new_zeros(checklist_target.size())
            initial_checklist_states.append(ChecklistStatelet(terminal_actions=terminal_actions,
                                                              checklist_target=checklist_target,
                                                              checklist_mask=checklist_mask,
                                                              checklist=initial_checklist))
        initial_state = CoverageState(batch_indices=list(range(batch_size)),
                                      action_history=[[] for _ in range(batch_size)],
                                      score=initial_score_list,
                                      rnn_state=initial_rnn_state,
                                      grammar_state=initial_grammar_state,
                                      possible_actions=actions,
                                      extras=label_strings,
                                      checklist_state=initial_checklist_states)

        agenda_data = [agenda_[:, 0].cpu().data for agenda_ in agenda_list]
        outputs = self._decoder_trainer.decode(initial_state,  # type: ignore
                                               self._decoder_step,
                                               partial(self._get_state_cost, worlds))
        if identifier is not None:
            outputs['identifier'] = identifier
        best_final_states = outputs['best_final_states']
        best_action_sequences = {}
        for batch_index, states in best_final_states.items():
            best_action_sequences[batch_index] = [state.action_history[0] for state in states]
        batch_action_strings = self._get_action_strings(actions, best_action_sequences)
        batch_denotations = self._get_denotations(batch_action_strings, worlds)
        if labels is not None:
            # We're either training or validating.
            self._update_metrics(action_strings=batch_action_strings,
                                 worlds=worlds,
                                 label_strings=label_strings,
                                 possible_actions=actions,
                                 agenda_data=agenda_data)
        else:
            # We're testing.
            outputs["best_action_strings"] = batch_action_strings
            outputs["denotations"] = batch_denotations
        return outputs

    def _get_checklist_info(self,
                            agenda: torch.LongTensor,
                            all_actions: List[ProductionRule]) -> Tuple[torch.Tensor,
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
        ``all_actions`` : ``List[ProductionRule]``
            All actions for one instance.
        """
        terminal_indices = []
        target_checklist_list = []
        agenda_indices_set = set([int(x) for x in agenda.squeeze(0).detach().cpu().numpy()])
        for index, action in enumerate(all_actions):
            # Each action is a ProductionRule, a tuple where the first item is the production
            # rule string.
            if action[0] in self._terminal_productions:
                terminal_indices.append([index])
                if index in agenda_indices_set:
                    target_checklist_list.append([1])
                else:
                    target_checklist_list.append([0])
        # We want to return checklist target and terminal actions that are column vectors to make
        # computing softmax over the difference between checklist and target easier.
        # (num_terminals, 1)
        terminal_actions = agenda.new_tensor(terminal_indices)
        # (num_terminals, 1)
        target_checklist = agenda.new_tensor(target_checklist_list, dtype=torch.float)
        if self._penalize_non_agenda_actions:
            # All terminal actions are relevant
            checklist_mask = torch.ones_like(target_checklist)
        else:
            checklist_mask = (target_checklist != 0).float()
        return target_checklist, terminal_actions, checklist_mask

    def _update_metrics(self,
                        action_strings: List[List[List[str]]],
                        worlds: List[List[NlvrWorld]],
                        label_strings: List[List[str]],
                        possible_actions: List[List[ProductionRule]],
                        agenda_data: List[List[int]]) -> None:
        # TODO(pradeep): Move this to the base class.
        # TODO(pradeep): action_strings contains k-best lists. This method only uses the top decoded
        # sequence currently. Maybe define top-k metrics?
        batch_size = len(worlds)
        for i in range(batch_size):
            # Using only the top decoded sequence per instance.
            instance_action_strings = action_strings[i][0] if action_strings[i] else []
            sequence_is_correct = [False]
            in_agenda_ratio = 0.0
            instance_possible_actions = possible_actions[i]
            if instance_action_strings:
                terminal_agenda_actions = []
                for rule_id in agenda_data[i]:
                    if rule_id == -1:
                        continue
                    action_string = instance_possible_actions[rule_id][0]
                    right_side = action_string.split(" -> ")[1]
                    if right_side.isdigit() or ('[' not in right_side and len(right_side) > 1):
                        terminal_agenda_actions.append(action_string)
                actions_in_agenda = [action in instance_action_strings for action in
                                     terminal_agenda_actions]
                in_agenda_ratio = sum(actions_in_agenda) / len(actions_in_agenda)
                instance_label_strings = label_strings[i]
                instance_worlds = worlds[i]
                sequence_is_correct = self._check_denotation(instance_action_strings,
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

    def _get_state_cost(self, batch_worlds: List[List[NlvrWorld]], state: CoverageState) -> torch.Tensor:
        """
        Return the cost of a finished state. Since it is a finished state, the group size will be
        1, and hence we'll return just one cost.

        The ``batch_worlds`` parameter here is because we need the world to check the denotation
        accuracy of the action sequence in the finished state.  Instead of adding a field to the
        ``State`` object just for this method, we take the ``World`` as a parameter here.
        """
        if not state.is_finished():
            raise RuntimeError("_get_state_cost() is not defined for unfinished states!")
        instace_worlds = batch_worlds[state.batch_indices[0]]
        # Our checklist cost is a sum of squared error from where we want to be, making sure we
        # take into account the mask.
        checklist_balance = state.checklist_state[0].get_balance()
        checklist_cost = torch.sum((checklist_balance) ** 2)

        # This is the number of items on the agenda that we want to see in the decoded sequence.
        # We use this as the denotation cost if the path is incorrect.
        # Note: If we are penalizing the model for producing non-agenda actions, this is not the
        # upper limit on the checklist cost. That would be the number of terminal actions.
        denotation_cost = torch.sum(state.checklist_state[0].checklist_target.float())
        checklist_cost = self._checklist_cost_weight * checklist_cost
        # TODO (pradeep): The denotation based cost below is strict. May be define a cost based on
        # how many worlds the logical form is correct in?
        # extras being None happens when we are testing. We do not care about the cost
        # then.  TODO (pradeep): Make this cleaner.
        if state.extras is None or all(self._check_state_denotations(state, instace_worlds)):
            cost = checklist_cost
        else:
            cost = checklist_cost + (1 - self._checklist_cost_weight) * denotation_cost
        return cost

    def _get_state_info(self,
                        state: CoverageState,
                        batch_worlds: List[List[NlvrWorld]]) -> Dict[str, List]:
        """
        This method is here for debugging purposes, in case you want to look at the what the model
        is learning. It may be inefficient to call it while training the model on real data.
        """
        if len(state.batch_indices) == 1 and state.is_finished():
            costs = [float(self._get_state_cost(batch_worlds, state).detach().cpu().numpy())]
        else:
            costs = []
        model_scores = [float(score.detach().cpu().numpy()) for score in state.score]
        all_actions = state.possible_actions[0]
        action_sequences = [[self._get_action_string(all_actions[action]) for action in history]
                            for history in state.action_history]
        agenda_sequences = []
        all_agenda_indices = []
        for checklist_state in state.checklist_state:
            agenda_indices = []
            for action, is_wanted in zip(checklist_state.terminal_actions, checklist_state.checklist_target):
                action_int = int(action.detach().cpu().numpy())
                is_wanted_int = int(is_wanted.detach().cpu().numpy())
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
