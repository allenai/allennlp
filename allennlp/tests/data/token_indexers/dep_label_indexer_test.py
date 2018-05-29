# pylint: disable=no-self-use,invalid-name
from collections import defaultdict

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token, Vocabulary
from allennlp.data.token_indexers import DepLabelIndexer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter


class TestDepLabelIndexer(AllenNlpTestCase):
    def setUp(self):
        super(TestDepLabelIndexer, self).setUp()
        self.tokenizer = SpacyWordSplitter(parse=True)

    def test_count_vocab_items_uses_pos_tags(self):
        tokens = self.tokenizer.split_words("This is a sentence.")
        tokens = [Token("<S>")] + [t for t in tokens] + [Token("</S>")]
        indexer = DepLabelIndexer()
        counter = defaultdict(lambda: defaultdict(int))
        for token in tokens:
            indexer.count_vocab_items(token, counter)

        assert counter["dep_labels"] == {"ROOT": 1, "nsubj": 1,
                                         "det": 1, "NONE": 2, "attr": 1, "punct": 1}

    def test_token_to_indices_uses_pos_tags(self):
        tokens = self.tokenizer.split_words("This is a sentence.")
        tokens = [t for t in tokens] + [Token("</S>")]
        vocab = Vocabulary()
        root_index = vocab.add_token_to_namespace('ROOT', namespace='dep_labels')
        none_index = vocab.add_token_to_namespace('NONE', namespace='dep_labels')
        indexer = DepLabelIndexer()
        assert indexer.token_to_indices(tokens[1], vocab) == root_index
        assert indexer.token_to_indices(tokens[-1], vocab) == none_index

    def test_padding_functions(self):
        indexer = DepLabelIndexer()
        assert indexer.get_padding_token() == 0
        assert indexer.get_padding_lengths(0) == {}

    def test_as_array_produces_token_sequence(self):
        indexer = DepLabelIndexer()
        padded_tokens = indexer.pad_token_sequence([1, 2, 3, 4, 5], 10, {})
        assert padded_tokens == [1, 2, 3, 4, 5, 0, 0, 0, 0, 0]
