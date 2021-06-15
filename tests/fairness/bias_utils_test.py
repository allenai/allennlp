import json
import torch

from allennlp.fairness.bias_utils import load_words, load_word_pairs

from allennlp.common.file_utils import cached_path
from allennlp.common.testing.test_case import AllenNlpTestCase
from allennlp.data import Instance, Token
from allennlp.data.batch import Batch
from allennlp.data import Vocabulary
from allennlp.data.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.fields import TextField


class BiasUtilsTest(AllenNlpTestCase):
    def setup_method(self):
        token_indexer = SingleIdTokenIndexer("tokens")

        self.pairs_fname = (
            "https://raw.githubusercontent.com/tolga-b/debiaswe/"
            "4c3fa843ffff45115c43fe112d4283c91d225c09/data/definitional_pairs.json"
        )
        with open(cached_path(self.pairs_fname)) as f:
            pairs_list = []
            [
                pairs_list.extend(
                    [w1.lower(), w2.lower(), w1.title(), w2.title(), w1.upper(), w2.upper()]
                )
                for w1, w2 in json.load(f)
            ]

        text_field = TextField(
            [Token(t) for t in pairs_list],
            {"tokens": token_indexer},
        )
        instance = Instance({"text": text_field})
        dataset = Batch([instance])
        self.pairs_vocab = Vocabulary.from_instances(dataset)
        self.num_pairs = len(set(pairs_list))

        self.singles_fname = (
            "https://raw.githubusercontent.com/tolga-b/debiaswe/"
            "4c3fa843ffff45115c43fe112d4283c91d225c09/data/gender_specific_full.json"
        )
        with open(cached_path(self.singles_fname)) as f:
            singles_list = json.load(f)

        text_field = TextField(
            [Token(t) for t in singles_list],
            {"tokens": token_indexer},
        )
        instance = Instance({"text": text_field})
        dataset = Batch([instance])
        self.singles_vocab = Vocabulary.from_instances(dataset)
        self.num_singles = len(set(singles_list))

        super().setup_method()

    def test_load_word_pairs(self):
        ids1, ids2 = load_word_pairs(
            self.pairs_fname, WhitespaceTokenizer(), self.pairs_vocab, "tokens"
        )
        # first two token IDs reserved for [CLS] and [SEP]
        assert torch.equal(
            torch.tensor([i.item() for i in ids1]), torch.arange(2, self.num_pairs + 2, step=2)
        )
        assert torch.equal(
            torch.tensor([i.item() for i in ids2]), torch.arange(3, self.num_pairs + 3, step=2)
        )

    def test_load_words(self):
        ids = load_words(
            self.singles_fname, WhitespaceTokenizer(), self.singles_vocab, "tokens", all_cases=False
        )
        # first two token IDs reserved for [CLS] and [SEP]
        assert torch.equal(
            torch.tensor([i.item() for i in ids]), torch.arange(2, self.num_singles + 2)
        )
