# pylint: disable=no-self-use,invalid-name
from collections import defaultdict

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token, Vocabulary
from allennlp.data.token_indexers import PosTagIndexer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter


class TestPosTagIndexer(AllenNlpTestCase):
    def setUp(self):
        super(TestPosTagIndexer, self).setUp()
        self.tokenizer = SpacyWordSplitter(pos_tags=True)

    def test_count_vocab_items_uses_pos_tags(self):
        tokens = self.tokenizer.split_words("This is a sentence.")
        tokens = [Token("<S>")] + [t for t in tokens] + [Token("</S>")]
        indexer = PosTagIndexer()
        counter = defaultdict(lambda: defaultdict(int))
        for token in tokens:
            indexer.count_vocab_items(token, counter)
        assert counter["pos_tags"] == {'DT': 2, 'VBZ': 1, '.': 1, 'NN': 1, 'NONE': 2}

        indexer._coarse_tags = True  # pylint: disable=protected-access
        counter = defaultdict(lambda: defaultdict(int))
        for token in tokens:
            indexer.count_vocab_items(token, counter)
        assert counter["pos_tags"] == {'VERB': 1, 'PUNCT': 1, 'DET': 2, 'NOUN': 1, 'NONE': 2}

    def test_token_to_indices_uses_pos_tags(self):
        tokens = self.tokenizer.split_words("This is a sentence.")
        tokens = [t for t in tokens] + [Token("</S>")]
        vocab = Vocabulary()
        verb_index = vocab.add_token_to_namespace('VERB', namespace='pos_tags')
        cop_index = vocab.add_token_to_namespace('VBZ', namespace='pos_tags')
        none_index = vocab.add_token_to_namespace('NONE', namespace='pos_tags')
        indexer = PosTagIndexer(coarse_tags=True)
        assert indexer.token_to_indices(tokens[1], vocab) == verb_index
        assert indexer.token_to_indices(tokens[-1], vocab) == none_index
        indexer._coarse_tags = False  # pylint: disable=protected-access
        assert indexer.token_to_indices(tokens[1], vocab) == cop_index

    def test_padding_functions(self):
        indexer = PosTagIndexer()
        assert indexer.get_padding_token() == 0
        assert indexer.get_padding_lengths(0) == {}

    def test_as_array_produces_token_sequence(self):
        indexer = PosTagIndexer()
        padded_tokens = indexer.pad_token_sequence([1, 2, 3, 4, 5], 10, {})
        assert padded_tokens == [1, 2, 3, 4, 5, 0, 0, 0, 0, 0]
