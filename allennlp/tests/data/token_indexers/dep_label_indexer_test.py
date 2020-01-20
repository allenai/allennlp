from collections import defaultdict

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token, Vocabulary
from allennlp.data.token_indexers import DepLabelIndexer
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer


class TestDepLabelIndexer(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.tokenizer = SpacyTokenizer(parse=True)

    def test_count_vocab_items_uses_pos_tags(self):
        tokens = self.tokenizer.tokenize("This is a sentence.")
        tokens = [Token("<S>")] + [t for t in tokens] + [Token("</S>")]
        indexer = DepLabelIndexer()
        counter = defaultdict(lambda: defaultdict(int))
        for token in tokens:
            indexer.count_vocab_items(token, counter)

        assert counter["dep_labels"] == {
            "ROOT": 1,
            "nsubj": 1,
            "det": 1,
            "NONE": 2,
            "attr": 1,
            "punct": 1,
        }

    def test_tokens_to_indices_uses_pos_tags(self):
        tokens = self.tokenizer.tokenize("This is a sentence.")
        tokens = [t for t in tokens] + [Token("</S>")]
        vocab = Vocabulary()
        root_index = vocab.add_token_to_namespace("ROOT", namespace="dep_labels")
        none_index = vocab.add_token_to_namespace("NONE", namespace="dep_labels")
        indexer = DepLabelIndexer()
        assert indexer.tokens_to_indices([tokens[1]], vocab) == {"tokens": [root_index]}
        assert indexer.tokens_to_indices([tokens[-1]], vocab) == {"tokens": [none_index]}

    def test_as_array_produces_token_sequence(self):
        indexer = DepLabelIndexer()
        padded_tokens = indexer.as_padded_tensor_dict({"tokens": [1, 2, 3, 4, 5]}, {"tokens": 10})
        assert padded_tokens["tokens"].tolist() == [1, 2, 3, 4, 5, 0, 0, 0, 0, 0]
