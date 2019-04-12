import logging
import os
from functools import partial
from typing import Dict, List, Tuple, Set, Any

from overrides import overrides
import torch

from allennlp.data import Vocabulary
from allennlp.data.fields.production_rule_field import ProductionRule
from allennlp.models.archival import load_archive, Archive
from allennlp.models.model import Model
from allennlp.models.semantic_parsing.wikitables.wikitables_semantic_parser import WikiTablesSemanticParser
from allennlp.modules import Attention, FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.semparse.type_declarations import wikitables_lambda_dcs as types
from allennlp.semparse.worlds import WikiTablesWorld
from allennlp.state_machines.states import CoverageState, ChecklistStatelet
from allennlp.state_machines.trainers import ExpectedRiskMinimization
from allennlp.state_machines.transition_functions import LinkingCoverageTransitionFunction
from allennlp.training.metrics import Average

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("wikitables_erm_parser")
class WikiTablesErmSemanticParser(WikiTablesSemanticParser):
    """
    A ``WikiTablesErmSemanticParser`` is a :class:`WikiTablesSemanticParser` that learns to search
    for logical forms that yield the correct denotations.

    Parameters
    ----------
    vocab : ``Vocabulary``
    question_embedder : ``TextFieldEmbedder``
        Embedder for questions. Passed to super class.
    action_embedding_dim : ``int``
        Dimension to use for action embeddings. Passed to super class.
    encoder : ``Seq2SeqEncoder``
        The encoder to use for the input question. Passed to super class.
    entity_encoder : ``Seq2VecEncoder``
        The encoder to used for averaging the words of an entity. Passed to super class.
    attention : ``Attention``
        We compute an attention over the input question at each step of the decoder, using the
        decoder hidden state as the query.  Passed to the transition function.
    decoder_beam_size : ``int``
        Beam size to be used by the ExpectedRiskMinimization algorithm.
    decoder_num_finished_states : ``int``
        Number of finished states for which costs will be computed by the ExpectedRiskMinimization
        algorithm.
    max_decoding_steps : ``int``
        Maximum number of steps the decoder should take before giving up. Used both during training
        and evaluation. Passed to super class.
    add_action_bias : ``bool``, optional (default=True)
        If ``True``, we will learn a bias weight for each action that gets used when predicting
        that action, in addition to its embedding.  Passed to super class.
    normalize_beam_score_by_length : ``bool``, optional (default=False)
        Should we normalize the log-probabilities by length before renormalizing the beam? This was
        shown to work better for NML by Edunov et al., but that many not be the case for semantic
        parsing.
    checklist_cost_weight : ``float``, optional (default=0.6)
        Mixture weight (0-1) for combining coverage cost and denotation cost. As this increases, we
        weigh the coverage cost higher, with a value of 1.0 meaning that we do not care about
        denotation accuracy.
    use_neighbor_similarity_for_linking : ``bool``, optional (default=False)
        If ``True``, we will compute a max similarity between a question token and the `neighbors`
        of an entity as a component of the linking scores.  This is meant to capture the same kind
        of information as the ``related_column`` feature. Passed to super class.
    dropout : ``float``, optional (default=0)
        If greater than 0, we will apply dropout with this probability after all encoders (pytorch
        LSTMs do not apply dropout to their last layer). Passed to super class.
    num_linking_features : ``int``, optional (default=10)
        We need to construct a parameter vector for the linking features, so we need to know how
        many there are.  The default of 10 here matches the default in the ``KnowledgeGraphField``,
        which is to use all ten defined features. If this is 0, another term will be added to the
        linking score. This term contains the maximum similarity value from the entity's neighbors
        and the question. Passed to super class.
    rule_namespace : ``str``, optional (default=rule_labels)
        The vocabulary namespace to use for production rules.  The default corresponds to the
        default used in the dataset reader, so you likely don't need to modify this. Passed to super
        class.
    tables_directory : ``str``, optional (default=/wikitables/)
        The directory to find tables when evaluating logical forms.  We rely on a call to SEMPRE to
        evaluate logical forms, and SEMPRE needs to read the table from disk itself.  This tells
        SEMPRE where to find the tables. Passed to super class.
    mml_model_file : ``str``, optional (default=None)
        If you want to initialize this model using weights from another model trained using MML,
        pass the path to the ``model.tar.gz`` file of that model here.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 question_embedder: TextFieldEmbedder,
                 action_embedding_dim: int,
                 encoder: Seq2SeqEncoder,
                 entity_encoder: Seq2VecEncoder,
                 attention: Attention,
                 decoder_beam_size: int,
                 decoder_num_finished_states: int,
                 max_decoding_steps: int,
                 mixture_feedforward: FeedForward = None,
                 add_action_bias: bool = True,
                 normalize_beam_score_by_length: bool = False,
                 checklist_cost_weight: float = 0.6,
                 use_neighbor_similarity_for_linking: bool = False,
                 dropout: float = 0.0,
                 num_linking_features: int = 10,
                 rule_namespace: str = 'rule_labels',
                 tables_directory: str = '/wikitables/',
                 mml_model_file: str = None) -> None:
        use_similarity = use_neighbor_similarity_for_linking
        super().__init__(vocab=vocab,
                         question_embedder=question_embedder,
                         action_embedding_dim=action_embedding_dim,
                         encoder=encoder,
                         entity_encoder=entity_encoder,
                         max_decoding_steps=max_decoding_steps,
                         add_action_bias=add_action_bias,
                         use_neighbor_similarity_for_linking=use_similarity,
                         dropout=dropout,
                         num_linking_features=num_linking_features,
                         rule_namespace=rule_namespace,
                         tables_directory=tables_directory)
        # Not sure why mypy needs a type annotation for this!
        self._decoder_trainer: ExpectedRiskMinimization = \
                ExpectedRiskMinimization(beam_size=decoder_beam_size,
                                         normalize_by_length=normalize_beam_score_by_length,
                                         max_decoding_steps=self._max_decoding_steps,
                                         max_num_finished_states=decoder_num_finished_states)
        unlinked_terminals_global_indices = []
        global_vocab = self.vocab.get_token_to_index_vocabulary(rule_namespace)
        for production, index in global_vocab.items():
            right_side = production.split(" -> ")[1]
            if right_side in types.COMMON_NAME_MAPPING:
                # This is a terminal production.
                unlinked_terminals_global_indices.append(index)
        self._num_unlinked_terminals = len(unlinked_terminals_global_indices)
        self._decoder_step = LinkingCoverageTransitionFunction(encoder_output_dim=self._encoder.get_output_dim(),
                                                               action_embedding_dim=action_embedding_dim,
                                                               input_attention=attention,
                                                               num_start_types=self._num_start_types,
                                                               predict_start_type_separately=True,
                                                               add_action_bias=self._add_action_bias,
                                                               mixture_feedforward=mixture_feedforward,
                                                               dropout=dropout)
        self._checklist_cost_weight = checklist_cost_weight
        self._agenda_coverage = Average()
        # TODO (pradeep): Checking whether file exists here to avoid raising an error when we've
        # copied a trained ERM model from a different machine and the original MML model that was
        # used to initialize it does not exist on the current machine. This may not be the best
        # solution for the problem.
        if mml_model_file is not None:
            if os.path.isfile(mml_model_file):
                archive = load_archive(mml_model_file)
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
        question_embedder_weight = "_question_embedder.token_embedder_tokens.weight"
        if question_embedder_weight not in archived_parameters or \
           question_embedder_weight not in model_parameters:
            raise RuntimeError("When initializing model weights from an MML model, we need "
                               "the question embedder to be a TokenEmbedder using namespace called "
                               "tokens.")
        for name, weights in archived_parameters.items():
            if name in model_parameters:
                if name == question_embedder_weight:
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
                question: Dict[str, torch.LongTensor],
                table: Dict[str, torch.LongTensor],
                world: List[WikiTablesWorld],
                actions: List[List[ProductionRule]],
                agenda: torch.LongTensor,
                example_lisp_string: List[str],
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        question : Dict[str, torch.LongTensor]
           The output of ``TextField.as_array()`` applied on the question ``TextField``. This will
           be passed through a ``TextFieldEmbedder`` and then through an encoder.
        table : ``Dict[str, torch.LongTensor]``
            The output of ``KnowledgeGraphField.as_array()`` applied on the table
            ``KnowledgeGraphField``.  This output is similar to a ``TextField`` output, where each
            entity in the table is treated as a "token", and we will use a ``TextFieldEmbedder`` to
            get embeddings for each entity.
        world : ``List[WikiTablesWorld]``
            We use a ``MetadataField`` to get the ``World`` for each input instance.  Because of
            how ``MetadataField`` works, this gets passed to us as a ``List[WikiTablesWorld]``,
        actions : ``List[List[ProductionRule]]``
            A list of all possible actions for each ``World`` in the batch, indexed into a
            ``ProductionRule`` using a ``ProductionRuleField``.  We will embed all of these
            and use the embeddings to determine which action to take at each timestep in the
            decoder.
        agenda : ``torch.LongTensor``
            Agenda of one instance of size ``(agenda_size, 1)``.
        example_lisp_string : ``List[str]``
            The example (lisp-formatted) string corresponding to the given input.  This comes
            directly from the ``.examples`` file provided with the dataset.  We pass this to SEMPRE
            when evaluating denotation accuracy; it is otherwise unused.
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            Metadata containing the original tokenized question within a 'question_tokens' key.
        """
        batch_size = list(question.values())[0].size(0)
        # Each instance's agenda is of size (agenda_size, 1)
        agenda_list = [agenda[i] for i in range(batch_size)]
        checklist_states = []
        all_terminal_productions = [set(instance_world.terminal_productions.values())
                                    for instance_world in world]
        max_num_terminals = max([len(terminals) for terminals in all_terminal_productions])
        for instance_actions, instance_agenda, terminal_productions in zip(actions,
                                                                           agenda_list,
                                                                           all_terminal_productions):
            checklist_info = self._get_checklist_info(instance_agenda,
                                                      instance_actions,
                                                      terminal_productions,
                                                      max_num_terminals)
            checklist_target, terminal_actions, checklist_mask = checklist_info
            initial_checklist = checklist_target.new_zeros(checklist_target.size())
            checklist_states.append(ChecklistStatelet(terminal_actions=terminal_actions,
                                                      checklist_target=checklist_target,
                                                      checklist_mask=checklist_mask,
                                                      checklist=initial_checklist))
        outputs: Dict[str, Any] = {}
        rnn_state, grammar_state = self._get_initial_rnn_and_grammar_state(question,
                                                                           table,
                                                                           world,
                                                                           actions,
                                                                           outputs)

        batch_size = len(rnn_state)
        initial_score = rnn_state[0].hidden_state.new_zeros(batch_size)
        initial_score_list = [initial_score[i] for i in range(batch_size)]
        initial_state = CoverageState(batch_indices=list(range(batch_size)),  # type: ignore
                                      action_history=[[] for _ in range(batch_size)],
                                      score=initial_score_list,
                                      rnn_state=rnn_state,
                                      grammar_state=grammar_state,
                                      checklist_state=checklist_states,
                                      possible_actions=actions,
                                      extras=example_lisp_string,
                                      debug_info=None)

        if not self.training:
            initial_state.debug_info = [[] for _ in range(batch_size)]

        outputs = self._decoder_trainer.decode(initial_state,  # type: ignore
                                               self._decoder_step,
                                               partial(self._get_state_cost, world))
        best_final_states = outputs['best_final_states']

        if not self.training:
            batch_size = len(actions)
            agenda_indices = [actions_[:, 0].cpu().data for actions_ in agenda]
            action_mapping = {}
            for batch_index, batch_actions in enumerate(actions):
                for action_index, action in enumerate(batch_actions):
                    action_mapping[(batch_index, action_index)] = action[0]
            for i in range(batch_size):
                in_agenda_ratio = 0.0
                # Decoding may not have terminated with any completed logical forms, if `num_steps`
                # isn't long enough (or if the model is not trained enough and gets into an
                # infinite action loop).
                if i in best_final_states:
                    action_sequence = best_final_states[i][0].action_history[0]
                    action_strings = [action_mapping[(i, action_index)] for action_index in action_sequence]
                    instance_possible_actions = actions[i]
                    agenda_actions = []
                    for rule_id in agenda_indices[i]:
                        rule_id = int(rule_id)
                        if rule_id == -1:
                            continue
                        action_string = instance_possible_actions[rule_id][0]
                        agenda_actions.append(action_string)
                    actions_in_agenda = [action in action_strings for action in agenda_actions]
                    if actions_in_agenda:
                        # Note: This means that when there are no actions on agenda, agenda coverage
                        # will be 0, not 1.
                        in_agenda_ratio = sum(actions_in_agenda) / len(actions_in_agenda)
                self._agenda_coverage(in_agenda_ratio)

            self._compute_validation_outputs(actions,
                                             best_final_states,
                                             world,
                                             example_lisp_string,
                                             metadata,
                                             outputs)
        return outputs

    @staticmethod
    def _get_checklist_info(agenda: torch.LongTensor,
                            all_actions: List[ProductionRule],
                            terminal_productions: Set[str],
                            max_num_terminals: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Takes an agenda, a list of all actions, a set of terminal productions in the corresponding
        world, and a length to pad the checklist vectors to, and returns a target checklist against
        which the checklist at each state will be compared to compute a loss, indices of
        ``terminal_actions``, and a ``checklist_mask`` that indicates which of the terminal actions
        are relevant for checklist loss computation.

        Parameters
        ----------
        ``agenda`` : ``torch.LongTensor``
            Agenda of one instance of size ``(agenda_size, 1)``.
        ``all_actions`` : ``List[ProductionRule]``
            All actions for one instance.
        ``terminal_productions`` : ``Set[str]``
            String representations of terminal productions in the corresponding world.
        ``max_num_terminals`` : ``int``
            Length to which the checklist vectors will be padded till. This is the max number of
            terminal productions in all the worlds in the batch.
        """
        terminal_indices = []
        target_checklist_list = []
        agenda_indices_set = set([int(x) for x in agenda.squeeze(0).detach().cpu().numpy()])
        # We want to return checklist target and terminal actions that are column vectors to make
        # computing softmax over the difference between checklist and target easier.
        for index, action in enumerate(all_actions):
            # Each action is a ProductionRule, a tuple where the first item is the production
            # rule string.
            if action[0] in terminal_productions:
                terminal_indices.append([index])
                if index in agenda_indices_set:
                    target_checklist_list.append([1])
                else:
                    target_checklist_list.append([0])
        while len(target_checklist_list) < max_num_terminals:
            target_checklist_list.append([0])
            terminal_indices.append([-1])
        # (max_num_terminals, 1)
        terminal_actions = agenda.new_tensor(terminal_indices)
        # (max_num_terminals, 1)
        target_checklist = agenda.new_tensor(target_checklist_list, dtype=torch.float)
        checklist_mask = (target_checklist != 0).float()
        return target_checklist, terminal_actions, checklist_mask

    def _get_state_cost(self, worlds: List[WikiTablesWorld], state: CoverageState) -> torch.Tensor:
        if not state.is_finished():
            raise RuntimeError("_get_state_cost() is not defined for unfinished states!")
        world = worlds[state.batch_indices[0]]

        # Our checklist cost is a sum of squared error from where we want to be, making sure we
        # take into account the mask. We clamp the lower limit of the balance at 0 to avoid
        # penalizing agenda actions produced multiple times.
        checklist_balance = torch.clamp(state.checklist_state[0].get_balance(), min=0.0)
        checklist_cost = torch.sum((checklist_balance) ** 2)

        # This is the number of items on the agenda that we want to see in the decoded sequence.
        # We use this as the denotation cost if the path is incorrect.
        denotation_cost = torch.sum(state.checklist_state[0].checklist_target.float())
        checklist_cost = self._checklist_cost_weight * checklist_cost
        action_history = state.action_history[0]
        batch_index = state.batch_indices[0]
        action_strings = [state.possible_actions[batch_index][i][0] for i in action_history]
        logical_form = world.get_logical_form(action_strings)
        lisp_string = state.extras[batch_index]
        if self._executor.evaluate_logical_form(logical_form, lisp_string):
            cost = checklist_cost
        else:
            cost = checklist_cost + (1 - self._checklist_cost_weight) * denotation_cost
        return cost

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        The base class returns a dict with dpd accuracy, denotation accuracy, and logical form
        percentage metrics. We add the agenda coverage metric here.
        """
        metrics = super().get_metrics(reset)
        metrics["agenda_coverage"] = self._agenda_coverage.get_metric(reset)
        return metrics
