from typing import Dict, List
import logging
import gzip
import pyparsing

from overrides import overrides

import tqdm

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset import Dataset
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.fields import TextField, KnowledgeGraphField, LabelField, ListField
from allennlp.data.dataset_readers.wikitables import TableKnowledgeGraph, World
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.seq2seq import START_SYMBOL, END_SYMBOL

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("wikitables")
class WikitablesDatasetReader(DatasetReader):
    """
    Reader for WikitableQuestions (https://github.com/ppasupat/WikiTableQuestions/releases/tag/v1.0.2).
    We assume you are reading in ``data/*.examples`` files, and you have access to the output from
    Dynamic Programming on Denotations (DPD) on the training dataset.

    Parameters
    ----------
    tables_directory : ``str``
        Prefix for the path to the directory in which the tables reside. For example, ``*.examples`` files
        contain paths like ``csv/204-csv/590.csv``, this is the directory that contains the ``csv`` directory.
    dpd_output_directory : ``str``
        Directory that contains all the gzipped dpd output files. We assume the filenames match the
        example IDs (e.g.: ``nt-0.gz``).
    utterance_tokenizer : ``Tokenizer`` (optional)
        Tokenizer to use for the questions. Will default to ``WordTokenizer()``.
    utterance_token_indexers : ``Dict[str, TokenIndexer]`` (optional)
        Token indexers for questions. Will default to ``{"tokens": SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 tables_directory: str,
                 dpd_output_directory: str = None,
                 utterance_tokenizer: Tokenizer = None,
                 utterance_token_indexers: Dict[str, TokenIndexer] = None) -> None:
        self._tables_directory = tables_directory
        self._dpd_output_directory = dpd_output_directory
        self._utterance_tokenizer = utterance_tokenizer or WordTokenizer()
        self._utterance_token_indexers = utterance_token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def read(self, file_path):
        instances = []
        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file: %s", file_path)
            for line in tqdm.tqdm(data_file):
                line = line.strip("\n")
                if not line:
                    continue
                instances.append(self.text_to_instance(line, True))
        if not instances:
            raise ConfigurationError("No instances read!")
        return Dataset(instances)

    @overrides
    def text_to_instance(self,
                         input_lisp_string: str,
                         for_train: bool = False) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        parsed_info = self._parse_line_as_lisp(input_lisp_string)
        tokenized_utterance = self._utterance_tokenizer.tokenize(parsed_info["utterance"])
        utterance_field = TextField(tokenized_utterance, self._utterance_token_indexers)
        table_filename = "%s/%s" % (self._tables_directory, parsed_info["context"])
        # We want the TSV file, but the ``*.examples`` files typically point to CSV.
        table_filename = table_filename.replace(".csv", ".tsv")
        table_knowledge_graph = TableKnowledgeGraph.read_from_file(table_filename)
        table_field = KnowledgeGraphField(table_knowledge_graph, self._utterance_token_indexers)
        if for_train:
            world = World(table_knowledge_graph)
            dpd_output_filename = "%s/%s.gz" % (self._dpd_output_directory, parsed_info["id"])
            sempre_forms = [line.strip().decode('utf-8') for line in gzip.open(dpd_output_filename)]
            expressions = world.process_sempre_forms(sempre_forms)
            action_sequences = [world.get_action_sequence(expression) for expression in expressions]
            action_sequences_field = ListField([self._make_action_sequence_field(sequence)
                                                for sequence in action_sequences])
            return Instance({"utterance": utterance_field,
                             "context": table_field,
                             "action_sequences": action_sequences_field})
        else:
            return Instance({"utterance": utterance_field, "context": table_field})

    @staticmethod
    def _make_action_sequence_field(action_sequence: List[str]) -> ListField:
        action_sequence.insert(0, START_SYMBOL)
        action_sequence.append(END_SYMBOL)
        return ListField([LabelField(action, label_namespace='actions') for action in action_sequence])

    @staticmethod
    def _parse_line_as_lisp(lisp_string: str) -> Dict[str, List[str]]:
        parsed_info = {x: None for x in ["id", "utterance", "context", "targetValue"]}
        input_nested_list = pyparsing.OneOrMore(pyparsing.nestedExpr()).parseString(lisp_string).asList()[0]
        for key_value in input_nested_list:
            if not isinstance(key_value, list):
                continue
            if key_value[0] == "id":
                parsed_info["id"] = key_value[1]
            elif key_value[0] == "utterance":
                parsed_info["utterance"] = key_value[1].replace("\"", "")
            elif key_value[0] == "context":
                parsed_info["context"] = key_value[1][-1]
            elif key_value[0] == "targetValue":
                parsed_info["targetValue"] = [x[1] for x in key_value[1][1:]]
        assert all([parsed_info[x] is not None for x in ["id", "utterance", "context"]]), "Invalid format"
        return parsed_info

    @classmethod
    def from_params(cls, params: Params) -> 'WikitablesDatasetReader':
        tables_directory = params.pop('tables_directory')
        dpd_output_directory = params.pop('dpd_output_directory', None)
        utterance_tokenizer_type = params.pop('utterance_tokenizer', None)
        if utterance_tokenizer_type:
            utterance_tokenizer = Tokenizer.from_params(utterance_tokenizer_type)
        else:
            utterance_tokenizer = None
        utterance_token_indexers_type = params.pop('utterance_token_indexers', None)
        if utterance_token_indexers_type:
            utterance_token_indexers = TokenIndexer.dict_from_params(utterance_token_indexers_type)
        else:
            utterance_token_indexers = None
        params.assert_empty(cls.__name__)
        return WikitablesDatasetReader(tables_directory,
                                       dpd_output_directory,
                                       utterance_tokenizer,
                                       utterance_token_indexers)
