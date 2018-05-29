import logging
import json
from typing import List, Dict
from collections import defaultdict

from overrides import overrides
import tqdm

from allennlp.common import Params
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.fields import Field, TextField, ProductionRuleField, LabelField, ListField
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.semparse.worlds import NlvrWorld


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("agenda-prediction")
class AgendaPredictionDatasetReader(DatasetReader):
    """
    ``DatasetReader`` for an agenda prediction model. The data format we expect is the same as the
    data read for semantic parsing models trained using Maximum Marginal Likelihood. For now, we
    assume that the domain is NLVR, but we will soon support WikiTables as well.

    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.
    tokenizer : ``Tokenizer`` (optional)
        The tokenizer used for sentences in NLVR. Default is ``WordTokenizer``
    sentence_token_indexers : ``Dict[str, TokenIndexer]`` (optional)
        Token indexers for tokens in input sentences.
        Default is ``{"tokens": SingleIdTokenIndexer()}``
    add_nonterminal_productions : ``bool``, optional (default=False)
        Should we add nonterminal productions to the target sequences? The answer is yes if you are
        training the agenda predictor to predict non-terminal actions as well. It is set to False by
        default.
    max_num_actions_in_agenda : ``int``, optional (default=5)
        The action sequences we read are typically noisy. So it is a good idea to train a model to
        predict only the top-k most frequent actions in the given sequences. This is the value of k,
        and it is 5 by default.
    """
    # TODO (pradeep): Support WikiTables as well.
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 sentence_token_indexers: Dict[str, TokenIndexer] = None,
                 add_nonterminal_productions: bool = False,
                 max_num_actions_in_agenda: int = 5) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._sentence_token_indexers = sentence_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._add_nonterminal_productions = add_nonterminal_productions
        self._max_num_actions_in_agenda = max_num_actions_in_agenda

    @overrides
    def _read(self, file_path: str):
        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file: %s", file_path)
            for line in tqdm.tqdm(data_file):
                line = line.strip("\n")
                if not line:
                    continue
                data = json.loads(line)
                sentence = data["sentence"]
                target_sequences = data["correct_sequences"]
                instance = self.text_to_instance(sentence,
                                                 target_sequences)
                if instance is not None:
                    yield instance

    @overrides
    def text_to_instance(self,  # type: ignore
                         sentence: str,
                         target_sequences: List[List[str]] = None) -> Instance:
        """
        Parameters
        ----------
        sentence : ``str``
            The query sentence.
        target_sequences : ``List[List[str]]`` (optional)
            List of target action sequences that are decompositions of logical forms conrresponding
            to the sentence.
        """
        # pylint: disable=arguments-differ
        tokenized_sentence = self._tokenizer.tokenize(sentence)
        sentence_field = TextField(tokenized_sentence, self._sentence_token_indexers)
        # TODO (pradeep): Hard-coding NLVR.
        # Instantiating an empty NLVR world for getting actions.
        world = NlvrWorld([])
        production_rule_fields: List[Field] = []
        considered_actions = []
        for action in world.all_possible_actions():
            if self._add_nonterminal_productions or world.is_terminal(action.split(' -> ')[1]):
                considered_actions.append(action)
                field = ProductionRuleField(action, is_global_rule=True)
                production_rule_fields.append(field)
        action_field = ListField(production_rule_fields)
        fields = {"sentence": sentence_field,
                  "all_actions": action_field}
        if target_sequences:
            action_frequencies: Dict[str, int] = defaultdict(int)
            for sequence in target_sequences:
                for action in sequence:
                    if self._add_nonterminal_productions or world.is_terminal(action.split(' -> ')[1]):
                        action_frequencies[action] += 1
            sorted_actions = sorted([(frequency, action) for action, frequency in
                                     action_frequencies.items()], reverse=True)
            frequent_actions = {action for _, action in
                                sorted_actions[:self._max_num_actions_in_agenda]}
            target_action_fields: List[Field] = []
            for action in considered_actions:
                if action in frequent_actions:
                    target_action_fields.append(LabelField("in"))
                else:
                    target_action_fields.append(LabelField("out"))
            fields["target_actions"] = ListField(target_action_fields)
        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'AgendaPredictionDatasetReader':
        lazy = params.pop('lazy', False)
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        sentence_token_indexers = TokenIndexer.dict_from_params(params.pop('sentence_token_indexers', {}))
        add_nonterminals = params.pop("add_nonterminals", False)
        max_num_actions = params.pop_int("max_num_actions", 5)
        params.assert_empty(cls.__name__)
        return cls(lazy=lazy,
                   tokenizer=tokenizer,
                   sentence_token_indexers=sentence_token_indexers,
                   add_nonterminal_productions=add_nonterminals,
                   max_num_actions_in_agenda=max_num_actions)
