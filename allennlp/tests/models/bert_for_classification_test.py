# pylint: disable=abstract-method
from typing import Dict

from pytorch_pretrained_bert.modeling import BertConfig, BertModel
from _pytest.monkeypatch import MonkeyPatch

from allennlp.common.testing import ModelTestCase
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import WordTokenizer


@DatasetReader.register("bert_classification_test")
class TestReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: WordTokenizer = None) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {}
        self.tokenizer = tokenizer or WordTokenizer()

    def _read(self, file_path: str):
        #            2   3    4   3     5     6   8      9    2   14   12
        sentence1 = "the quickest quick brown fox jumped over the lazy dog"
        tokens1 = self.tokenizer.tokenize(sentence1)
        label1 = "positive"

        #            2   3     5     6   8      9    2  15 10 11 14   1
        sentence2 = "the quick brown fox jumped over the laziest lazy elmo"
        tokens2 = self.tokenizer.tokenize(sentence2)
        label2 = "negative"

        instance1 = Instance({"tokens": TextField(tokens1, self._token_indexers),
                              "label": LabelField(label1)})
        instance2 = Instance({"tokens": TextField(tokens2, self._token_indexers),
                              "label": LabelField(label2)})

        return [instance1, instance2]


class TestBertForClassification(ModelTestCase):
    def setUp(self):
        super().setUp()
        monkeypatch = MonkeyPatch()

        # monkeypatch BertModel.from_pretrained to return the tiny test fixture model
        config_path = self.FIXTURES_ROOT / 'bert' / 'config.json'
        config = BertConfig(str(config_path))
        monkeypatch.setattr(BertModel, 'from_pretrained', lambda _: BertModel(config))

    def test_model_can_train_save_and_load(self):
        param_file = self.FIXTURES_ROOT / 'bert' / 'bert_for_classification.jsonnet'

        self.set_up_model(param_file, "")
        self.ensure_model_can_train_save_and_load(param_file)

    def test_bert_pooler_can_train_save_and_load(self):
        param_file = self.FIXTURES_ROOT / 'bert' / 'bert_pooler.jsonnet'

        self.set_up_model(param_file, "")
        self.ensure_model_can_train_save_and_load(param_file)
