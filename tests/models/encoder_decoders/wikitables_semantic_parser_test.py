# pylint: disable=invalid-name
from typing import List

import tqdm

from allennlp.common.testing import ModelTestCase
from allennlp.common import Params
from allennlp.data import Dataset, DatasetReader, Instance, Token
from allennlp.data.fields import Field, LabelField, ListField, TextField
from allennlp.data.tokenizers import CharacterTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.dataset_readers.seq2seq import START_SYMBOL, END_SYMBOL


@DatasetReader.register("toy_wikitables")
class ToyWikiTablesDatasetReader(DatasetReader):
    def __init__(self):
        self._tokenizer = CharacterTokenizer()
        self._token_indexers = {'tokens': SingleIdTokenIndexer()}

    def read(self, file_path):
        instances = []
        with open(file_path, "r") as data_file:
            for line in tqdm.tqdm(data_file):
                line = line.strip("\n")
                line_parts = line.split('\t')
                source_sequence, targets = line_parts[0], line_parts[1:]
                instances.append(self.text_to_instance(source_sequence, targets))
        return Dataset(instances)

    def text_to_instance(self,  # type: ignore
                         source_string: str,
                         targets: List[str] = None) -> Instance:
        # pylint: disable=arguments-differ
        tokenized_source = self._tokenizer.tokenize(source_string)
        source_field = TextField(tokenized_source, self._token_indexers)
        if targets is not None:
            target_field: Field = ListField([self._make_target_field(target)
                                             for target in targets])
            return Instance({"question": source_field, "target_action_sequences": target_field})
        else:
            return Instance({'question': source_field})

    def _make_target_field(self, target_string) -> ListField:
        tokenized_target = self._tokenizer.tokenize(target_string)
        tokenized_target.insert(0, Token(START_SYMBOL))
        tokenized_target.append(Token(END_SYMBOL))
        return ListField([LabelField(target_token.text, label_namespace='actions')
                          for target_token in tokenized_target])

    @classmethod
    def from_params(cls, params: Params) -> 'ToyWikiTablesDatasetReader':
        params.assert_empty(cls.__name__)
        return ToyWikiTablesDatasetReader()


class WikiTablesSemanticParserTest(ModelTestCase):
    def setUp(self):
        super(WikiTablesSemanticParserTest, self).setUp()
        self.set_up_model("tests/fixtures/encoder_decoder/wikitables_semantic_parser/experiment.json",
                          "tests/fixtures/data/seq2seq_max_marginal_likelihood.tsv")

    def test_encoder_decoder_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
