from typing import Dict

from pytorch_pretrained_bert.modeling import BertConfig, BertModel
from _pytest.monkeypatch import MonkeyPatch

from allennlp.common.testing import ModelTestCase
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertModel


@DatasetReader.register("bert_classification_test")
class BertClassificationTestReader(DatasetReader):
    def __init__(
        self,
        lazy: bool = False,
        token_indexers: Dict[str, TokenIndexer] = None,
        tokenizer: WordTokenizer = None,
    ) -> None:
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

        instance1 = Instance(
            {"tokens": TextField(tokens1, self._token_indexers), "label": LabelField(label1)}
        )
        instance2 = Instance(
            {"tokens": TextField(tokens2, self._token_indexers), "label": LabelField(label2)}
        )

        return [instance1, instance2]


class TestBertForClassification(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.monkeypatch = MonkeyPatch()

        # monkeypatch the PretrainedBertModel to return the tiny test fixture model
        config_path = self.FIXTURES_ROOT / "bert" / "config.json"
        config = BertConfig(str(config_path))
        self.monkeypatch.setattr(BertModel, "from_pretrained", lambda _: BertModel(config))

    def tearDown(self):
        self.monkeypatch.undo()
        super().tearDown()

    def test_model_can_train_save_and_load(self):
        param_file = self.FIXTURES_ROOT / "bert" / "bert_for_classification.jsonnet"

        self.set_up_model(param_file, "")
        self.ensure_model_can_train_save_and_load(param_file)

    def test_decode(self):
        param_file = self.FIXTURES_ROOT / "bert" / "bert_for_classification.jsonnet"
        self.set_up_model(param_file, "")
        padding_lengths = self.dataset.get_padding_lengths()
        tensors = self.model(**self.dataset.as_tensor_dict(padding_lengths))
        decoded = self.model.decode(tensors)
        assert "label" in decoded

    def test_bert_pooler_can_train_save_and_load(self):
        param_file = self.FIXTURES_ROOT / "bert" / "bert_pooler.jsonnet"

        self.set_up_model(param_file, "")
        self.ensure_model_can_train_save_and_load(param_file)

    def test_caching(self):
        model1 = PretrainedBertModel.load("testing caching")
        model2 = PretrainedBertModel.load("testing caching")
        assert model1 is model2

        model3 = PretrainedBertModel.load("testing not caching", cache_model=False)
        model4 = PretrainedBertModel.load("testing not caching", cache_model=False)
        assert model3 is not model4

        model5 = PretrainedBertModel.load("name1")
        model6 = PretrainedBertModel.load("name2")
        assert model5 is not model6
