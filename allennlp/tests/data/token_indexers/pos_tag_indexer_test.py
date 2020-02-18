from collections import defaultdict

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token, Vocabulary
from allennlp.data.token_indexers import PosTagIndexer
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer


class TestPosTagIndexer(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.tokenizer = SpacyTokenizer(pos_tags=True)

    def test_count_vocab_items_uses_pos_tags(self):
        tokens = self.tokenizer.tokenize("This is a sentence.")
        tokens = [Token("<S>")] + [t for t in tokens] + [Token("</S>")]
        # Hard-coding this because spacy's POS tagger keeps changing on us, wanting to call this AUX
        # in some runs.
        tokens[2] = Token("is", tag_="VBZ", pos_="VERB")
        indexer = PosTagIndexer()
        counter = defaultdict(lambda: defaultdict(int))
        for token in tokens:
            indexer.count_vocab_items(token, counter)
        assert counter["pos_tokens"] == {"DT": 2, "VBZ": 1, ".": 1, "NN": 1, "NONE": 2}

        indexer._coarse_tags = True
        counter = defaultdict(lambda: defaultdict(int))
        for token in tokens:
            indexer.count_vocab_items(token, counter)
        assert counter["pos_tokens"] == {"VERB": 1, "PUNCT": 1, "DET": 2, "NOUN": 1, "NONE": 2}

    def test_tokens_to_indices_uses_pos_tags(self):
        tokens = self.tokenizer.tokenize("This is a sentence.")
        tokens = [t for t in tokens] + [Token("</S>")]
        tokens[1] = Token("is", tag_="VBZ", pos_="VERB")
        vocab = Vocabulary()
        verb_index = vocab.add_token_to_namespace("VERB", namespace="pos_tags")
        cop_index = vocab.add_token_to_namespace("VBZ", namespace="pos_tags")
        none_index = vocab.add_token_to_namespace("NONE", namespace="pos_tags")
        # Have to add other tokens too, since we're calling `tokens_to_indices` on all of them
        vocab.add_token_to_namespace("DET", namespace="pos_tags")
        vocab.add_token_to_namespace("NOUN", namespace="pos_tags")
        vocab.add_token_to_namespace("PUNCT", namespace="pos_tags")

        indexer = PosTagIndexer(namespace="pos_tags", coarse_tags=True)

        indices = indexer.tokens_to_indices(tokens, vocab)
        assert len(indices) == 1
        assert "tokens" in indices
        assert indices["tokens"][1] == verb_index
        assert indices["tokens"][-1] == none_index

        indexer._coarse_tags = False
        assert indexer.tokens_to_indices([tokens[1]], vocab) == {"tokens": [cop_index]}

    def test_as_array_produces_token_sequence(self):
        indexer = PosTagIndexer()
        padded_tokens = indexer.as_padded_tensor_dict({"tokens": [1, 2, 3, 4, 5]}, {"tokens": 10})
        assert padded_tokens["tokens"].tolist() == [1, 2, 3, 4, 5, 0, 0, 0, 0, 0]

    def test_blank_pos_tag(self):
        tokens = [Token(token) for token in "allennlp is awesome .".split(" ")]
        indexer = PosTagIndexer()
        counter = defaultdict(lambda: defaultdict(int))
        for token in tokens:
            indexer.count_vocab_items(token, counter)
        # spacy uses a empty string to indicate "no POS tag"
        # we convert it to "NONE"
        assert counter["pos_tokens"]["NONE"] == 4
        vocab = Vocabulary(counter)
        none_index = vocab.get_token_index("NONE", "pos_tokens")
        # should raise no exception
        indices = indexer.tokens_to_indices(tokens, vocab)
        assert {"tokens": [none_index, none_index, none_index, none_index]} == indices
