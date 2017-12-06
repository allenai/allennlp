from typing import Dict, List
import json
import logging

from overrides import overrides
import tqdm

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import JsonDict
from allennlp.data.instance import Instance
from allennlp.data.fields import TextField, ListField, LabelField
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.dataset import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.semparse.type_declarations import nlvr_type_declaration as nlvr_types


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("nlvr")
class NlvrDatasetReader(DatasetReader):
    """
    ``DatasetReader`` for the NLVR domain.
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 sentence_token_indexers: Dict[str, TokenIndexer] = None) -> None:
        self._tokenizer = tokenizer or WordTokenizer()
        self._sentence_token_indexers = sentence_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._agenda_mapping = {}
        for constant in nlvr_types.COMMON_NAME_MAPPING:
            alias = nlvr_types.COMMON_NAME_MAPPING[constant]
            if alias not in nlvr_types.COMMON_TYPE_SIGNATURE:
                continue
            if alias in nlvr_types.COMMON_TYPE_SIGNATURE:
                constant_type = nlvr_types.COMMON_TYPE_SIGNATURE[alias]
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
                data = json.loads(line)
                sentence = data["sentence"]
                label = data["label"]
                structured_rep = data["structured_rep"]
                instances.append(self.text_to_instance(sentence, structured_rep, label))
        if not instances:
            raise ConfigurationError("No instances read!")
        return Dataset(instances)

    @overrides
    def text_to_instance(self,  # type: ignore
                         sentence: str,
                         structured_rep: JsonDict,
                         label: str = None) -> Instance:
        """
        TODO
        """
        # pylint: disable=arguments-differ,unused-argument
        # TODO(pradeep): Use a knowledgegraph field or a new field for the structured rep.
        tokenized_sentence = self._tokenizer.tokenize(sentence)
        sentence_field = TextField(tokenized_sentence, self._sentence_token_indexers)
        agenda = self._get_agenda_for_sentence(sentence)
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
        agenda = []
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
