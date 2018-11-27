from typing import Any, Dict, List

from overrides import overrides
import torch

from allennlp.data import Vocabulary
from allennlp.data.fields.production_rule_field import ProductionRule
from allennlp.models.model import Model
from allennlp.models.semantic_parsing.wikitables.wikitables_semantic_parser import WikiTablesSemanticParser
from allennlp.modules import Attention, FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.semparse.worlds import WikiTablesWorld
from allennlp.state_machines import BeamSearch
from allennlp.state_machines.states import GrammarBasedState
from allennlp.state_machines.trainers import MaximumMarginalLikelihood
from allennlp.state_machines.transition_functions import LinkingTransitionFunction


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
    attention : ``Attention``
        We compute an attention over the input question at each step of the decoder, using the
        decoder hidden state as the query.  Passed to the transition function.
    mixture_feedforward : ``FeedForward``, optional (default=None)
        If given, we'll use this to compute a mixture probability between global actions and linked
        actions given the hidden state at every timestep of decoding, instead of concatenating the
        logits for both (where the logits may not be compatible with each other).  Passed to
        the transition function.
    add_action_bias : ``bool``, optional (default=True)
        If ``True``, we will learn a bias weight for each action that gets used when predicting
        that action, in addition to its embedding.  Passed to super class.
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
                 decoder_beam_search: BeamSearch,
                 max_decoding_steps: int,
                 attention: Attention,
                 mixture_feedforward: FeedForward = None,
                 add_action_bias: bool = True,
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
                         add_action_bias=add_action_bias,
                         use_neighbor_similarity_for_linking=use_similarity,
                         dropout=dropout,
                         num_linking_features=num_linking_features,
                         rule_namespace=rule_namespace,
                         tables_directory=tables_directory)
        self._beam_search = decoder_beam_search
        self._decoder_trainer = MaximumMarginalLikelihood(training_beam_size)
        self._decoder_step = LinkingTransitionFunction(encoder_output_dim=self._encoder.get_output_dim(),
                                                       action_embedding_dim=action_embedding_dim,
                                                       input_attention=attention,
                                                       num_start_types=self._num_start_types,
                                                       predict_start_type_separately=True,
                                                       add_action_bias=self._add_action_bias,
                                                       mixture_feedforward=mixture_feedforward,
                                                       dropout=dropout)

    @overrides
    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                table: Dict[str, torch.LongTensor],
                world: List[WikiTablesWorld],
                actions: List[List[ProductionRule]],
                example_lisp_string: List[str] = None,
                target_action_sequences: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
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
        actions : ``List[List[ProductionRule]]``
            A list of all possible actions for each ``World`` in the batch, indexed into a
            ``ProductionRule`` using a ``ProductionRuleField``.  We will embed all of these
            and use the embeddings to determine which action to take at each timestep in the
            decoder.
        example_lisp_string : ``List[str]``, optional (default = None)
            The example (lisp-formatted) string corresponding to the given input.  This comes
            directly from the ``.examples`` file provided with the dataset.  We pass this to SEMPRE
            when evaluating denotation accuracy; it is otherwise unused.
        target_action_sequences : torch.Tensor, optional (default = None)
           A list of possibly valid action sequences, where each action is an index into the list
           of possible actions.  This tensor has shape ``(batch_size, num_action_sequences,
           sequence_length)``.
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            Metadata containing the original tokenized question within a 'question_tokens' key.
        """
        outputs: Dict[str, Any] = {}
        rnn_state, grammar_state = self._get_initial_rnn_and_grammar_state(question,
                                                                           table,
                                                                           world,
                                                                           actions,
                                                                           outputs)
        batch_size = len(rnn_state)
        initial_score = rnn_state[0].hidden_state.new_zeros(batch_size)
        initial_score_list = [initial_score[i] for i in range(batch_size)]
        initial_state = GrammarBasedState(batch_indices=list(range(batch_size)),  # type: ignore
                                          action_history=[[] for _ in range(batch_size)],
                                          score=initial_score_list,
                                          rnn_state=rnn_state,
                                          grammar_state=grammar_state,
                                          possible_actions=actions,
                                          extras=example_lisp_string,
                                          debug_info=None)

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

            self._compute_validation_outputs(actions,
                                             best_final_states,
                                             world,
                                             example_lisp_string,
                                             metadata,
                                             outputs)

            return outputs
