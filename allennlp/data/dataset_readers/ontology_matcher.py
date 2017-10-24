from typing import Dict, List
import logging

from overrides import overrides
import json
import tqdm

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.fields import Field, TextField, BooleanField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.instance import Instance
from allennlp.data.dataset import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("ontology_matcher")
class OntologyMatchingDatasetReader(DatasetReader):
    """
    Reads instances from a jsonlines file where each line is in the following format:

    {"match": X, "source": {kb_entity}, "target: {kb_entity}}
     X in [0, 1]
     kb_entity is a slightly modified KBEntity in json with fields:
        canonical_name
        aliases
        definition
        other_contexts
        relationships

    and converts it into a ``Dataset`` suitable for ontology matching.

    Parameters
    ----------
    token_delimiter: ``str``, optional (default=``None``)
        The text that separates each WORD-TAG pair from the next pair. If ``None``
        then the line will just be split on whitespace.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
        Note that the `output` tags will always correspond to single token IDs based on how they
        are pre-tokenised in the data file.
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 name_token_indexers: Dict[str, TokenIndexer] = None) -> None:
        self._name_token_indexers = name_token_indexers or \
                               {'tokens': SingleIdTokenIndexer(namespace="tokens"),
                                'token_characters': TokenCharactersIndexer(namespace="token_characters")}
        self._tokenizer = tokenizer or WordTokenizer()

    @overrides
    def read(self, file_path):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        instances = []

        # open data file and read lines
        with open(file_path, 'r') as ontm_file:
            logger.info("Reading ontology matching instances from jsonl dataset at: %s", file_path)
            for line in tqdm.tqdm(ontm_file):
                training_pair = json.loads(line)
                s_ent = training_pair['source_ent']
                t_ent = training_pair['target_ent']
                label = training_pair['label']

                # convert entry to instance and append to instances
                instances.append(self.text_to_instance(s_ent, t_ent, label))

        if not instances:
            raise ConfigurationError("No instances were read from the given filepath {}. "
                                     "Is the path correct?".format(file_path))
        return Dataset(instances)

    @overrides
    def text_to_instance(self,  # type: ignore
                         s_ent: dict,
                         t_ent: dict,
                         label: str = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        # tokenize names
        s_name_tokens = self._tokenizer.tokenize(s_ent['canonical_name'].lower())
        t_name_tokens = self._tokenizer.tokenize(t_ent['canonical_name'].lower())

        # add entity name fields
        fields['s_ent_name'] = TextField(s_name_tokens, self._name_token_indexers)
        fields['t_ent_name'] = TextField(t_name_tokens, self._name_token_indexers)

        # add boolean label (0 = no match, 1 = match)
        fields['label'] = BooleanField(label)

        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'OntologyMatchingDatasetReader':
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        name_token_indexers = TokenIndexer.dict_from_params(params.pop('name_token_indexers', {}))
        params.assert_empty(cls.__name__)
        return OntologyMatchingDatasetReader(tokenizer=tokenizer,
                                             name_token_indexers=name_token_indexers)
