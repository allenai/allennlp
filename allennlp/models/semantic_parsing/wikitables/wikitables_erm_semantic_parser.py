from typing import Dict, List

from overrides import overrides
import torch
from torch.autograd import Variable

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.data.fields.production_rule_field import ProductionRuleArray
from allennlp.models.model import Model
from allennlp.models.semantic_parsing.wikitables.wikitables_decoder_state import WikiTablesDecoderState
from allennlp.models.semantic_parsing.wikitables.wikitables_semantic_parser import WikiTablesSemanticParser
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, FeedForward
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.nn.decoding.decoder_trainers import ExpectedRiskMinimization
from allennlp.semparse.worlds import WikiTablesWorld


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
    attention_function : ``SimilarityFunction``
        We compute an attention over the input question at each step of the decoder, using the
        decoder hidden state as the query.  This is the similarity function we use for that
        attention. Passed to super class.
    decoder_beam_size : ``int``
        Beam size to be used by the ExpectedRiskMinimization algorithm.
    max_decoding_steps : ``int``
        Maximum number of steps the decoder should take before giving up. Used both during training
        and evaluation. Passed to super class.
    normalize_beam_score_by_length : ``bool``, optional (default=False)
        Should we normalize the log-probabilities by length before renormalizing the beam? This was
        shown to work better for NML by Edunov et al., but that many not be the case for semantic
        parsing.
    use_neighbor_similarity_for_linking : ``bool``, optional (default=False)
        If ``True``, we will compute a max similarity between a question token and the `neighbors`
        of an entity as a component of the linking scores.  This is meant to capture the same kind
        of information as the ``related_column`` feature. Passed to super class.
    dropout : ``float``, optional (default=0)
        If greater than 0, we will apply dropout with this probability after all encoders (pytorch
        LSTMs do not apply dropout to their last layer). Passed to super class.
    num_linking_features : ``int``, optional (default=8)
        We need to construct a parameter vector for the linking features, so we need to know how
        many there are.  The default of 8 here matches the default in the ``KnowledgeGraphField``,
        which is to use all eight defined features. If this is 0, another term will be added to the
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
                 attention_function: SimilarityFunction,
                 decoder_beam_size: int,
                 max_decoding_steps: int,
                 normalize_beam_score_by_length: bool = False,
                 use_neighbor_similarity_for_linking: bool = False,
                 dropout: float = 0.0,
                 num_linking_features: int = 8,
                 rule_namespace: str = 'rule_labels',
                 tables_directory: str = '/wikitables/') -> None:
        use_similarity = use_neighbor_similarity_for_linking
        super(WikiTablesErmSemanticParser, self).__init__(vocab=vocab,
                                                          question_embedder=question_embedder,
                                                          action_embedding_dim=action_embedding_dim,
                                                          encoder=encoder,
                                                          entity_encoder=entity_encoder,
                                                          mixture_feedforward=mixture_feedforward,
                                                          max_decoding_steps=max_decoding_steps,
                                                          attention_function=attention_function,
                                                          use_neighbor_similarity_for_linking=use_similarity,
                                                          dropout=dropout,
                                                          num_linking_features=num_linking_features,
                                                          rule_namespace=rule_namespace,
                                                          tables_directory=tables_directory)
        # Not sure why mypy needs a type annotation for this!
        self._decoder_trainer: ExpectedRiskMinimization = \
                ExpectedRiskMinimization(beam_size=decoder_beam_size,
                                         normalize_by_length=normalize_beam_score_by_length,
                                         max_decoding_steps=self._max_decoding_steps)

    @overrides
    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                table: Dict[str, torch.LongTensor],
                world: List[WikiTablesWorld],
                actions: List[List[ProductionRuleArray]],
                example_lisp_string: List[str]) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        # pylint: disable=unused-argument
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
        actions : ``List[List[ProductionRuleArray]]``
            A list of all possible actions for each ``World`` in the batch, indexed into a
            ``ProductionRuleArray`` using a ``ProductionRuleField``.  We will embed all of these
            and use the embeddings to determine which action to take at each timestep in the
            decoder.
        example_lisp_string : ``List[str]``
            The example (lisp-formatted) string corresponding to the given input.  This comes
            directly from the ``.examples`` file provided with the dataset.  We pass this to SEMPRE
            when evaluating denotation accuracy; it is otherwise unused.
        target_action_sequences : torch.Tensor, optional (default=None)
           A list of possibly valid action sequences, where each action is an index into the list
           of possible actions.  This tensor has shape ``(batch_size, num_action_sequences,
           sequence_length)``.
        """
        initial_info = self._get_initial_state_and_scores(question=question,
                                                          table=table,
                                                          world=world,
                                                          actions=actions,
                                                          example_lisp_string=example_lisp_string,
                                                          add_world_to_initial_state=True)
        initial_state = initial_info["initial_state"]
        outputs = self._decoder_trainer.decode(initial_state,
                                               self._decoder_step,
                                               self._get_state_cost)
        return outputs

    def _get_state_cost(self, state: WikiTablesDecoderState) -> torch.Tensor:
        if not state.is_finished():
            raise RuntimeError("_get_state_cost() is not defined for unfinished states!")
        action_history = state.action_history[0]
        batch_index = state.batch_indices[0]
        action_strings = [state.possible_actions[batch_index][i][0] for i in action_history]
        logical_form = state.world[batch_index].get_logical_form(action_strings)
        lisp_string = state.example_lisp_string[0]
        if self._denotation_accuracy.evaluate_logical_form(logical_form, lisp_string):
            cost = torch.FloatTensor([0.0])
        else:
            cost = torch.FloatTensor([10])
        return Variable(state.flattened_linking_scores.data.new(cost)).float()

    @classmethod
    def from_params(cls, vocab, params: Params) -> 'WikiTablesErmSemanticParser':
        question_embedder = TextFieldEmbedder.from_params(vocab, params.pop("question_embedder"))
        action_embedding_dim = params.pop_int("action_embedding_dim")
        encoder = Seq2SeqEncoder.from_params(params.pop("encoder"))
        entity_encoder = Seq2VecEncoder.from_params(params.pop('entity_encoder'))
        mixture_feedforward_type = params.pop('mixture_feedforward', None)
        if mixture_feedforward_type is not None:
            mixture_feedforward = FeedForward.from_params(mixture_feedforward_type)
        else:
            mixture_feedforward = None
        # If no attention function is specified, we should not use attention, not attention with
        # default similarity function.
        attention_function_type = params.pop("attention_function", None)
        if attention_function_type is not None:
            attention_function = SimilarityFunction.from_params(attention_function_type)
        else:
            attention_function = None
        decoder_beam_size = params.pop_int("decoder_beam_size")
        max_decoding_steps = params.pop_int("max_decoding_steps")
        normalize_beam_score_by_length = params.pop("normalize_beam_score_by_length", False)
        use_neighbor_similarity_for_linking = params.pop_bool("use_neighbor_similarity_for_linking", False)
        dropout = params.pop_float('dropout', 0.0)
        num_linking_features = params.pop_int('num_linking_features', 8)
        tables_directory = params.pop('tables_directory', '/wikitables/')
        rule_namespace = params.pop('rule_namespace', 'rule_labels')
        params.assert_empty(cls.__name__)
        return cls(vocab,
                   question_embedder=question_embedder,
                   action_embedding_dim=action_embedding_dim,
                   encoder=encoder,
                   entity_encoder=entity_encoder,
                   mixture_feedforward=mixture_feedforward,
                   attention_function=attention_function,
                   decoder_beam_size=decoder_beam_size,
                   max_decoding_steps=max_decoding_steps,
                   normalize_beam_score_by_length=normalize_beam_score_by_length,
                   use_neighbor_similarity_for_linking=use_neighbor_similarity_for_linking,
                   dropout=dropout,
                   num_linking_features=num_linking_features,
                   tables_directory=tables_directory,
                   rule_namespace=rule_namespace)
