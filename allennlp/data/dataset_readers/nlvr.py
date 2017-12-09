from typing import Dict, List
import json
import logging
from collections import defaultdict

from overrides import overrides
import tqdm

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data.instance import Instance
from allennlp.data.fields import TextField, ListField, LabelField
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.dataset import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.semparse.type_declarations import type_declaration as types
from allennlp.data.semparse.type_declarations import nlvr_type_declaration as nlvr_types


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("nlvr")
class NlvrDatasetReader(DatasetReader):
    """
    ``DatasetReader`` for the NLVR domain. In addition to the usual methods for reading files and instances
    from text, this class contains a method for creating an agenda of actions that each sentence triggers.

    Parameters
    ----------
    tokenizer : ``Tokenizer``
        The tokenizer used for sentences in NLVR.
    token_indexers : ``Dict[str, TokenIndexer]``
        Token indexers for tokens in input sentences.
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 sentence_token_indexers: Dict[str, TokenIndexer] = None) -> None:
        self._tokenizer = tokenizer or WordTokenizer()
        self._sentence_token_indexers = sentence_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._agenda_mapping = defaultdict(list)
        for constant in nlvr_types.COMMON_NAME_MAPPING:
            alias = nlvr_types.COMMON_NAME_MAPPING[constant]
            if alias in nlvr_types.COMMON_TYPE_SIGNATURE:
                constant_type = nlvr_types.COMMON_TYPE_SIGNATURE[alias]
                if constant_type != types.ANY_TYPE:
                    self._agenda_mapping[constant] = "%s -> %s" % (constant_type, constant)

    @overrides
    def read(self, file_path):
        instances = []
        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file: %s", file_path)
            for line in tqdm.tqdm(data_file):
                line = line.strip("\n")
                if not line:
                    continue
                instances.append(self.text_to_instance(line))
        if not instances:
            raise ConfigurationError("No instances read!")
        return Dataset(instances)

    @overrides
    def text_to_instance(self, json_line: str) -> Instance:
        # pylint: disable=arguments-differ
        # TODO(pradeep): Use a ``KnowledgeGraphField`` or a new field for the structured rep, for an
        # interface with ``NlvrWorld`` to execute logical forms.
        data = json.loads(json_line)
        sentence = data["sentence"]
        label = data["label"] if "label" in data else None
        structured_rep = data["structured_rep"]  # pylint: disable=unused-variable
        tokenized_sentence = self._tokenizer.tokenize(sentence)
        sentence_field = TextField(tokenized_sentence, self._sentence_token_indexers)
        agenda = self._get_agenda_for_sentence(sentence)
        assert agenda, "No agenda found for sentence: %s" % sentence
        agenda_field = ListField([LabelField(action, label_namespace='actions') for action in agenda])
        fields = {"sentence": sentence_field, "agenda": agenda_field}
        if label:
            label_field = LabelField(label, label_namespace='denotations')
            fields["label"] = label_field
        return Instance(fields)

    def _get_agenda_for_sentence(self, sentence: str) -> List[str]:
        """
        Given a ``sentence``, return a list of actions it triggers. The model tries to include as many
        of these actions in the decoded sequences as possible.
        """
        # TODO(pradeep): Use approximate and substring matching as well.
        agenda = ['t']
        # This takes care of shapes, colors, top, bottom, big small etc.
        for constant, production in self._agenda_mapping.items():
            if constant in sentence:
                agenda.append(production)
        if sentence.startswith("There is a "):
            agenda.append(self._agenda_mapping["assert_greater_equals"])
        if "tower" in sentence or "box" in sentence or "grey" in sentence:
            # Ensuring box filtering function (filter_*) at top.
            agenda.append("t -> [<b,t>, b]")
        return agenda

    @classmethod
    def from_params(cls, params: Params) -> 'NlvrDatasetReader':
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        sentence_token_indexers = TokenIndexer.dict_from_params(params.pop('sentence_token_indexers', {}))
        params.assert_empty(cls.__name__)
        return NlvrDatasetReader(tokenizer=tokenizer,
                                 sentence_token_indexers=sentence_token_indexers)
