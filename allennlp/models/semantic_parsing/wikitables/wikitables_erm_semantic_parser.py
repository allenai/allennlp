import logging
import os
from typing import Dict, List, Tuple

from overrides import overrides
import torch
from torch.autograd import Variable

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.data.fields.production_rule_field import ProductionRuleArray
from allennlp.models.model import Model
from allennlp.models.archival import load_archive, Archive
from allennlp.models.semantic_parsing.wikitables.wikitables_decoder_state import WikiTablesDecoderState
from allennlp.models.semantic_parsing.wikitables.wikitables_semantic_parser import WikiTablesSemanticParser
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, FeedForward
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.nn.decoding.decoder_trainers import ExpectedRiskMinimization
from allennlp.semparse import ParsingError
from allennlp.semparse.worlds import WikiTablesWorld

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
    initial_mml_model_file : ``str``, optional (default=None)
        If you want to initialize this model using weights from another model trained using MML,
        pass the path to the ``model.tar.gz`` file of that model here.
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
                 num_linking_features: int = 10,
                 rule_namespace: str = 'rule_labels',
                 tables_directory: str = '/wikitables/',
                 initial_mml_model_file: str = None) -> None:
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
                actions: List[List[ProductionRuleArray]],
                example_lisp_string: List[str]) -> Dict[str, torch.Tensor]:
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
        actions : ``List[List[ProductionRuleArray]]``
            A list of all possible actions for each ``World`` in the batch, indexed into a
            ``ProductionRuleArray`` using a ``ProductionRuleField``.  We will embed all of these
            and use the embeddings to determine which action to take at each timestep in the
            decoder.
        example_lisp_string : ``List[str]``
            The example (lisp-formatted) string corresponding to the given input.  This comes
            directly from the ``.examples`` file provided with the dataset.  We pass this to SEMPRE
            when evaluating denotation accuracy; it is otherwise unused.
        """
        initial_info = self._get_initial_state_and_scores(question=question,
                                                          table=table,
                                                          world=world,
                                                          actions=actions,
                                                          example_lisp_string=example_lisp_string,
                                                          add_world_to_initial_state=True)
        initial_state = initial_info["initial_state"]
        # TODO(pradeep): Keep track of debug info. It's not straightforward currently because the
        # ERM's decode does not return the best states.
        outputs = self._decoder_trainer.decode(initial_state,
                                               self._decoder_step,
                                               self._get_state_cost)
        if not self.training:
            # TODO(pradeep): Can move most of this block to super class.
            linking_scores = initial_info["linking_scores"]
            feature_scores = initial_info["feature_scores"]
            similarity_scores = initial_info["similarity_scores"]
            batch_size = list(question.values())[0].size(0)
            action_mapping = {}
            for batch_index, batch_actions in enumerate(actions):
                for action_index, action in enumerate(batch_actions):
                    action_mapping[(batch_index, action_index)] = action[0]
            outputs['action_mapping'] = action_mapping
            outputs['entities'] = []
            outputs['linking_scores'] = linking_scores
            if feature_scores is not None:
                outputs['feature_scores'] = feature_scores
            outputs['similarity_scores'] = similarity_scores
            outputs['logical_form'] = []
            best_action_sequences = outputs['best_action_sequences']
            for i in range(batch_size):
                # Decoding may not have terminated with any completed logical forms, if `num_steps`
                # isn't long enough (or if the model is not trained enough and gets into an
                # infinite action loop).
                if i in best_action_sequences:
                    # Taking only the top action sequence.
                    best_action_sequence = best_action_sequences[i][0]
                    action_strings = [action_mapping[(i, action_index)] for action_index in best_action_sequence]
                    try:
                        self._has_logical_form(1.0)
                        logical_form = world[i].get_logical_form(action_strings, add_var_function=False)
                    except ParsingError:
                        self._has_logical_form(0.0)
                        logical_form = 'Error producing logical form'
                    if example_lisp_string:
                        self._denotation_accuracy(logical_form, example_lisp_string[i])
                    outputs['logical_form'].append(logical_form)
                    outputs['entities'].append(world[i].table_graph.entities)
                else:
                    outputs['logical_form'].append('')
                    self._has_logical_form(0.0)
                    if example_lisp_string:
                        self._denotation_accuracy(None, example_lisp_string[i])
        return outputs

    def _get_state_cost(self, state: WikiTablesDecoderState) -> torch.Tensor:
        if not state.is_finished():
            raise RuntimeError("_get_state_cost() is not defined for unfinished states!")
        action_history = state.action_history[0]
        batch_index = state.batch_indices[0]
        action_strings = [state.possible_actions[batch_index][i][0] for i in action_history]
        logical_form = state.world[batch_index].get_logical_form(action_strings)
        lisp_string = state.example_lisp_string[batch_index]
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
        num_linking_features = params.pop_int('num_linking_features', 10)
        tables_directory = params.pop('tables_directory', '/wikitables/')
        rule_namespace = params.pop('rule_namespace', 'rule_labels')
        mml_model_file = params.pop('mml_model_file', None)
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
                   rule_namespace=rule_namespace,
                   initial_mml_model_file=mml_model_file)
