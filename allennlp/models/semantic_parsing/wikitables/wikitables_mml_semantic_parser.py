from typing import Any, Dict, List

from overrides import overrides
import torch

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.data.fields.production_rule_field import ProductionRuleArray
from allennlp.models.model import Model
from allennlp.models.semantic_parsing.wikitables.wikitables_decoder_step import WikiTablesDecoderStep
from allennlp.models.semantic_parsing.wikitables.wikitables_semantic_parser import WikiTablesSemanticParser
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, FeedForward
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.nn.decoding import BeamSearch
from allennlp.nn.decoding.decoder_trainers import MaximumMarginalLikelihood
from allennlp.semparse.worlds import WikiTablesWorld
from allennlp.semparse import ParsingError


@Model.register("wikitables_mml_parser")
class WikiTablesMmlSemanticParser(WikiTablesSemanticParser):
    """
    A ``WikiTablesMmlSemanticParser`` is a :class:`WikiTablesSemanticParser` which is trained to
    maximize the marginal likelihood of an approximate set of logical forms which give the correct
    denotation. This is a re-implementation of the model used for the paper `Neural Semantic Parsing with Type
    Constraints for Semi-Structured Tables
    <https://www.semanticscholar.org/paper/Neural-Semantic-Parsing-with-Type-Constraints-for-Krishnamurthy-Dasigi/8c6f58ed0ebf379858c0bbe02c53ee51b3eb398a>`_,
    by Jayant Krishnamurthy, Pradeep Dasigi, and Matt Gardner (EMNLP 2017).

    WORK STILL IN PROGRESS.  We'll iteratively improve it until we've reproduced the performance of
    the original parser.

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
    decoder_beam_search : ``BeamSearch``
        When we're not training, this is how we will do decoding.
    max_decoding_steps : ``int``
        When we're decoding with a beam search, what's the maximum number of steps we should take?
        This only applies at evaluation time, not during training. Passed to super class.
    attention_function : ``SimilarityFunction``
        We compute an attention over the input question at each step of the decoder, using the
        decoder hidden state as the query.  This is the similarity function we use for that
        attention. Passed to super class.
    training_beam_size : ``int``, optional (default=None)
        If given, we will use a constrained beam search of this size during training, so that we
        use only the top ``training_beam_size`` action sequences according to the model in the MML
        computation.  If this is ``None``, we will use all of the provided action sequences in the
        MML computation.
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
    """
    def __init__(self,
                 vocab: Vocabulary,
                 question_embedder: TextFieldEmbedder,
                 action_embedding_dim: int,
                 encoder: Seq2SeqEncoder,
                 entity_encoder: Seq2VecEncoder,
                 mixture_feedforward: FeedForward,
                 decoder_beam_search: BeamSearch,
                 max_decoding_steps: int,
                 attention_function: SimilarityFunction,
                 training_beam_size: int = None,
                 use_neighbor_similarity_for_linking: bool = False,
                 dropout: float = 0.0,
                 num_linking_features: int = 10,
                 rule_namespace: str = 'rule_labels',
                 tables_directory: str = '/wikitables/') -> None:
        use_similarity = use_neighbor_similarity_for_linking
        super().__init__(vocab=vocab,
                         question_embedder=question_embedder,
                         action_embedding_dim=action_embedding_dim,
                         encoder=encoder,
                         entity_encoder=entity_encoder,
                         max_decoding_steps=max_decoding_steps,
                         use_neighbor_similarity_for_linking=use_similarity,
                         dropout=dropout,
                         num_linking_features=num_linking_features,
                         rule_namespace=rule_namespace,
                         tables_directory=tables_directory)
        self._beam_search = decoder_beam_search
        self._decoder_trainer = MaximumMarginalLikelihood(training_beam_size)
        self._decoder_step = WikiTablesDecoderStep(encoder_output_dim=self._encoder.get_output_dim(),
                                                   action_embedding_dim=action_embedding_dim,
                                                   attention_function=attention_function,
                                                   num_start_types=self._num_start_types,
                                                   num_entity_types=self._num_entity_types,
                                                   mixture_feedforward=mixture_feedforward,
                                                   dropout=dropout)

    @overrides
    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                table: Dict[str, torch.LongTensor],
                world: List[WikiTablesWorld],
                actions: List[List[ProductionRuleArray]],
                example_lisp_string: List[str] = None,
                target_action_sequences: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        In this method we encode the table entities, link them to words in the question, then
        encode the question. Then we set up the initial state for the decoder, and pass that
        state off to either a DecoderTrainer, if we're training, or a BeamSearch for inference,
        if we're not.

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
        actions : ``List[List[ProductionRuleArray]]``
            A list of all possible actions for each ``World`` in the batch, indexed into a
            ``ProductionRuleArray`` using a ``ProductionRuleField``.  We will embed all of these
            and use the embeddings to determine which action to take at each timestep in the
            decoder.
        example_lisp_string : ``List[str]``, optional (default=None)
            The example (lisp-formatted) string corresponding to the given input.  This comes
            directly from the ``.examples`` file provided with the dataset.  We pass this to SEMPRE
            when evaluating denotation accuracy; it is otherwise unused.
        target_action_sequences : torch.Tensor, optional (default=None)
           A list of possibly valid action sequences, where each action is an index into the list
           of possible actions.  This tensor has shape ``(batch_size, num_action_sequences,
           sequence_length)``.
        """
        initial_info = self._get_initial_state_and_scores(question, table, world, actions)
        initial_state = initial_info["initial_state"]
        linking_scores = initial_info["linking_scores"]
        feature_scores = initial_info["feature_scores"]
        similarity_scores = initial_info["similarity_scores"]
        batch_size = list(question.values())[0].size(0)
        if target_action_sequences is not None:
            # Remove the trailing dimension (from ListField[ListField[IndexField]]).
            target_action_sequences = target_action_sequences.squeeze(-1)
            target_mask = target_action_sequences != self._action_padding_index
        else:
            target_mask = None

        if self.training:
            return self._decoder_trainer.decode(initial_state,
                                                self._decoder_step,
                                                (target_action_sequences, target_mask))
        else:
            # TODO(pradeep): Most of the functionality in this black can be moved to the super
            # class.
            action_mapping = {}
            for batch_index, batch_actions in enumerate(actions):
                for action_index, action in enumerate(batch_actions):
                    action_mapping[(batch_index, action_index)] = action[0]
            outputs: Dict[str, Any] = {'action_mapping': action_mapping}
            if target_action_sequences is not None:
                outputs['loss'] = self._decoder_trainer.decode(initial_state,
                                                               self._decoder_step,
                                                               (target_action_sequences, target_mask))['loss']
            num_steps = self._max_decoding_steps
            # This tells the state to start keeping track of debug info, which we'll pass along in
            # our output dictionary.
            initial_state.debug_info = [[] for _ in range(batch_size)]
            best_final_states = self._beam_search.search(num_steps,
                                                         initial_state,
                                                         self._decoder_step,
                                                         keep_final_unfinished_states=False)
            outputs['best_action_sequence'] = []
            outputs['debug_info'] = []
            outputs['entities'] = []
            outputs['linking_scores'] = linking_scores
            if feature_scores is not None:
                outputs['feature_scores'] = feature_scores
            outputs['similarity_scores'] = similarity_scores
            outputs['logical_form'] = []
            for i in range(batch_size):
                # Decoding may not have terminated with any completed logical forms, if `num_steps`
                # isn't long enough (or if the model is not trained enough and gets into an
                # infinite action loop).
                if i in best_final_states:
                    best_action_indices = best_final_states[i][0].action_history[0]
                    if target_action_sequences is not None:
                        # Use a Tensor, not a Variable, to avoid a memory leak.
                        targets = target_action_sequences[i].data
                        sequence_in_targets = 0
                        sequence_in_targets = self._action_history_match(best_action_indices, targets)
                        self._action_sequence_accuracy(sequence_in_targets)
                    action_strings = [action_mapping[(i, action_index)] for action_index in best_action_indices]
                    try:
                        self._has_logical_form(1.0)
                        logical_form = world[i].get_logical_form(action_strings, add_var_function=False)
                    except ParsingError:
                        self._has_logical_form(0.0)
                        logical_form = 'Error producing logical form'
                    if example_lisp_string:
                        self._denotation_accuracy(logical_form, example_lisp_string[i])
                    outputs['best_action_sequence'].append(action_strings)
                    outputs['logical_form'].append(logical_form)
                    outputs['debug_info'].append(best_final_states[i][0].debug_info[0])  # type: ignore
                    outputs['entities'].append(world[i].table_graph.entities)
                else:
                    outputs['logical_form'].append('')
                    self._has_logical_form(0.0)
                    if example_lisp_string:
                        self._denotation_accuracy(None, example_lisp_string[i])
            return outputs

    @classmethod
    def from_params(cls, vocab, params: Params) -> 'WikiTablesMmlSemanticParser':
        question_embedder = TextFieldEmbedder.from_params(vocab, params.pop("question_embedder"))
        action_embedding_dim = params.pop_int("action_embedding_dim")
        encoder = Seq2SeqEncoder.from_params(params.pop("encoder"))
        entity_encoder = Seq2VecEncoder.from_params(params.pop('entity_encoder'))
        max_decoding_steps = params.pop_int("max_decoding_steps")
        mixture_feedforward_type = params.pop('mixture_feedforward', None)
        if mixture_feedforward_type is not None:
            mixture_feedforward = FeedForward.from_params(mixture_feedforward_type)
        else:
            mixture_feedforward = None
        decoder_beam_search = BeamSearch.from_params(params.pop("decoder_beam_search"))
        # If no attention function is specified, we should not use attention, not attention with
        # default similarity function.
        attention_function_type = params.pop("attention_function", None)
        if attention_function_type is not None:
            attention_function = SimilarityFunction.from_params(attention_function_type)
        else:
            attention_function = None
        training_beam_size = params.pop_int('training_beam_size', None)
        use_neighbor_similarity_for_linking = params.pop_bool('use_neighbor_similarity_for_linking', False)
        dropout = params.pop_float('dropout', 0.0)
        num_linking_features = params.pop_int('num_linking_features', 10)
        tables_directory = params.pop('tables_directory', '/wikitables/')
        rule_namespace = params.pop('rule_namespace', 'rule_labels')
        params.assert_empty(cls.__name__)
        return cls(vocab,
                   question_embedder=question_embedder,
                   action_embedding_dim=action_embedding_dim,
                   encoder=encoder,
                   entity_encoder=entity_encoder,
                   mixture_feedforward=mixture_feedforward,
                   decoder_beam_search=decoder_beam_search,
                   max_decoding_steps=max_decoding_steps,
                   attention_function=attention_function,
                   training_beam_size=training_beam_size,
                   use_neighbor_similarity_for_linking=use_neighbor_similarity_for_linking,
                   dropout=dropout,
                   num_linking_features=num_linking_features,
                   tables_directory=tables_directory,
                   rule_namespace=rule_namespace)
