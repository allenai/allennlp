from typing import Dict, List
import json
import logging

from overrides import overrides
import tqdm

from allennlp.common import Params
from allennlp.common.util import JsonDict
from allennlp.data.instance import Instance
from allennlp.data.fields import Field, TextField, ListField, IndexField, LabelField
from allennlp.data.fields import ProductionRuleField, MetadataField
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.semparse.worlds import NlvrWorld


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("nlvr")
class NlvrDatasetReader(DatasetReader):
    """
    ``DatasetReader`` for the NLVR domain. In addition to the usual methods for reading files and
    instances from text, this class contains a method for creating an agenda of actions that each
    sentence triggers.

    We process the either the original json version of the NLVR dataset
    (http://lic.nlp.cornell.edu/nlvr/) or the processed version where we group all the worlds that a
    sentence appears in here.
    Note that we deal with the structured representations of the synthetic images instead of the
    actual images themselves.
    The format of each line in the original jsonl file is
    ```
    "sentence": <sentence>,
    "label": <true/false>,
    "identifier": <id>,
    "evals": <dict containing all annotations>,
    "structured_rep": <list of three box representations, where each box is a list of object
    representation dicts, containing fields "x_loc", "y_loc", "color", "type", "size">
    ```
    We use the fields ``sentence``, ``label`` and ``structured_rep``.
    And the format of the processed files is
    ```
    "sentence": <sentence>,
    "labels": <list of labels corresponding to worlds the sentence appears in>
    "identifier": <id that is only the prefix from the original data>
    "worlds": <list of structured representations>
    ```

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
    nonterminal_indexers : ``Dict[str, TokenIndexer]`` (optional)
        Indexers for non-terminals in production rules. The default is to index terminals and
        non-terminals in the same way, but you may want to change it.
        Default is ``{"tokens": SingleIdTokenIndexer("rule_labels")}``
    terminal_indexers : ``Dict[str, TokenIndexer]`` (optional)
        Indexers for terminals in production rules. The default is to index terminals and
        non-terminals in the same way, but you may want to change it.
        Default is ``{"tokens": SingleIdTokenIndexer("rule_labels")}``
    add_paths_to_agenda : ``bool`` (optional)
        If set to true, the agenda will also contain the non-terminal productions in the path from
        the root node to each of the desired terminal productions. We do an approximate heuristic
        search while computing the paths to avoid infinitely long ones (containing cycles).
    """
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 sentence_token_indexers: Dict[str, TokenIndexer] = None,
                 nonterminal_indexers: Dict[str, TokenIndexer] = None,
                 terminal_indexers: Dict[str, TokenIndexer] = None,
                 add_paths_to_agenda: bool = True) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._sentence_token_indexers = sentence_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._nonterminal_indexers = nonterminal_indexers or {"tokens":
                                                              SingleIdTokenIndexer("rule_labels")}
        self._terminal_indexers = terminal_indexers or {"tokens": SingleIdTokenIndexer("rule_labels")}
        self._add_paths_to_agenda = add_paths_to_agenda

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
                if "worlds" in data:
                    # This means that we are reading grouped nlvr data. There will be multiple
                    # worlds and corresponding labels per sentence.
                    labels = data["labels"]
                    structured_representations = data["worlds"]
                else:
                    # We will make lists of labels and structured representations, each with just
                    # one element for consistency.
                    labels = [data["label"]]
                    structured_representations = [data["structured_rep"]]
                yield self.text_to_instance(sentence, structured_representations, labels)

    @overrides
    def text_to_instance(self,  # type: ignore
                         sentence: str,
                         structured_representations: List[List[List[JsonDict]]],
                         labels: List[str] = None) -> Instance:
        """
        Parameters
        ----------
        sentence : ``str``
            The query sentence.
        structured_representations : ``List[List[List[JsonDict]]]``
            A list of Json representations of all the worlds. See expected format in this class' docstring.
        labels : ``List[str]`` (optional)
            List of string representations of the labels (true or false) corresponding to the
            ``structured_representations``. Not required while testing.
        """
        # pylint: disable=arguments-differ
        worlds = [NlvrWorld(data) for data in structured_representations]
        tokenized_sentence = self._tokenizer.tokenize(sentence)
        sentence_field = TextField(tokenized_sentence, self._sentence_token_indexers)
        # TODO(pradeep): Assuming every world gives the same agenda for a sentence. This is true
        # now, but may change later.
        agenda = worlds[0].get_agenda_for_sentence(sentence, self._add_paths_to_agenda)
        assert agenda, "No agenda found for sentence: %s" % sentence
        production_rule_fields: List[Field] = []
        instance_action_ids: Dict[str, int] = {}
        # TODO(pradeep): Assuming that possible actions are the same in all worlds. This may change
        # later too.
        for production_rule in worlds[0].all_possible_actions():
            instance_action_ids[production_rule] = len(instance_action_ids)
            field = ProductionRuleField(production_rule,
                                        terminal_indexers=self._terminal_indexers,
                                        nonterminal_indexers=self._nonterminal_indexers,
                                        is_nonterminal=lambda x: x not in worlds[0].terminal_productions,
                                        context=tokenized_sentence)
            production_rule_fields.append(field)
        action_field = ListField(production_rule_fields)
        # agenda_field contains indices into actions.
        agenda_field = ListField([IndexField(instance_action_ids[action], action_field)
                                  for action in agenda])
        worlds_field = ListField([MetadataField(world) for world in worlds])
        fields = {"sentence": sentence_field,
                  "agenda": agenda_field,
                  "worlds": worlds_field,
                  "actions": action_field}
        if labels:
            labels_field = ListField([LabelField(label, label_namespace='denotations') for label in
                                      labels])
            fields["labels"] = labels_field
        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'NlvrDatasetReader':
        lazy = params.pop('lazy', False)
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        sentence_token_indexers = TokenIndexer.dict_from_params(params.pop('sentence_token_indexers', {}))
        terminal_indexers = TokenIndexer.dict_from_params(params.pop('terminal_indexers', {}))
        nonterminal_indexers = TokenIndexer.dict_from_params(params.pop('nonterminal_indexers', {}))
        add_paths_to_agenda = params.pop("add_paths_to_agenda", True)
        params.assert_empty(cls.__name__)
        return NlvrDatasetReader(lazy=lazy,
                                 tokenizer=tokenizer,
                                 sentence_token_indexers=sentence_token_indexers,
                                 terminal_indexers=terminal_indexers,
                                 nonterminal_indexers=nonterminal_indexers,
                                 add_paths_to_agenda=add_paths_to_agenda)
