# pylint: disable=no-self-use,invalid-name,protected-access
import json
import tarfile
import spacy

import pytest

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token
from allennlp.data.token_indexers import OpenaiTransformerBytePairIndexer
from allennlp.data.token_indexers.openai_transformer_byte_pair_indexer import text_standardize
from allennlp.data.vocabulary import Vocabulary


class TestOpenaiTransformerBytePairIndexer(AllenNlpTestCase):
    def setUp(self):
        super().setUp()

        encoder_path = self.TEST_DIR / 'encoder.json'
        bpe_path = self.TEST_DIR / 'vocab.bpe'
        transformer_model_path = self.TEST_DIR / 'model.tar.gz'

        symbols = ["e", "w", "o", "wo", "."]
        byte_pairs = [(sym1, sym2 + end)
                      for sym1 in symbols        # prefer earlier first symbol
                      for sym2 in symbols        # if tie, prefer earlier second symbol
                      for end in ('</w>', '')]   # if tie, prefer ending a word
        encoding = {f"{sym1}{sym2}": idx + 1 for idx, (sym1, sym2) in enumerate(byte_pairs)}


        with open(encoder_path, 'w') as encoder_file:
            json.dump(encoding, encoder_file)

        with open(bpe_path, 'w') as bpe_file:
            bpe_file.write("#version 0.0\n")
            for sym1, sym2 in byte_pairs:
                bpe_file.write(f"{sym1} {sym2}\n")
            bpe_file.write("\n")

        with tarfile.open(transformer_model_path, 'w') as tf:
            tf.add(encoder_path, 'model/encoder_bpe_40000.json')
            tf.add(bpe_path, 'model/vocab_40000.bpe')

        self.indexer = OpenaiTransformerBytePairIndexer(encoding, byte_pairs)
        self.vocab = Vocabulary(non_padded_namespaces=['openai_transformer'])

    def test_bpe(self):

        # [e, w, o, e</w>] -> best pair (e, w)
        # [ew, o, e</w>] -> best pair (o, e</w>)
        # [ew, oe</w>] -> done
        token = Token("ewoe")
        assert self.indexer.byte_pair_encode(token) == ['ew', 'oe</w>']

        # Prefer "ew" to "we"
        token = Token("ewe")
        assert self.indexer.byte_pair_encode(token) == ['ew', 'e</w>']

        # Prefer ending a word
        token = Token("eee")
        assert self.indexer.byte_pair_encode(token) == ['e', 'ee</w>']

        # Encodes up to a single symbol when appropriate
        token = Token("woe")
        assert self.indexer.byte_pair_encode(token) == ['woe</w>']

    def test_tokens_to_indices(self):
        tokens = [Token('ewoe'), Token('woe'), Token('ewe'), Token('ee')]

        # vocab should be empty initially
        assert 'openai_transformer' not in self.vocab._index_to_token
        assert 'openai_transformer' not in self.vocab._token_to_index

        indices = self.indexer.tokens_to_indices(tokens, self.vocab, 'test')

        # vocab should be full now
        i2t = self.vocab._index_to_token.get('openai_transformer')
        t2i = self.vocab._token_to_index.get('openai_transformer')
        assert len(i2t) == 5 * 5 * 2
        assert len(t2i) == 5 * 5 * 2

        assert set(indices.keys()) == {"test", "test-offsets", "mask"}

        text_tokens = indices['test']
        offsets = indices['test-offsets']

        assert text_tokens[:6] == [
                self.indexer.encoder.get(symbol, 0)
                for symbol in ['ew', 'oe</w>'] + ['woe</w>'] + ['ew', 'e</w>'] + ['ee</w>']
        ]

        assert offsets == [
                1,  # end of first word
                2,  # end of second word
                4,  # end of third word
                5,  # end of last word
        ]

    def test_raises_with_too_long_sentence(self):
        tokens = [Token('a') for _ in range(513)]

        with pytest.raises(RuntimeError):
            self.indexer.tokens_to_indices(tokens, self.vocab, 'should-fail')

    @pytest.mark.skip()
    def test_for_correctness_with_fixture(self):
        bpe_path = "https://s3-us-west-2.amazonaws.com/allennlp/models/openai-transformer-lm-2018.07.23.tar.gz"
        indexer = OpenaiTransformerBytePairIndexer(model_path=bpe_path)

        with open(self.FIXTURES_ROOT / 'openai_transformer' / 'text.txt', 'r') as fin:
            sentences = fin.read().strip().split('\n')
        with open(self.FIXTURES_ROOT / 'openai_transformer' / 'indexed_text.json', 'r') as fin:
            expected_indices = json.load(fin)

        # tokenize and check that indices are correct
        nlp = spacy.load('en_core_web_sm')

        for k, sentence in enumerate(sentences):
            tokens = [token.text for token in nlp(text_standardize(sentence)) if not token.is_space]
            indices = indexer.tokens_to_indices(
                    [Token(token) for token in tokens], Vocabulary(), 'openai_indexer'
            )
            non_padded_indices = [i for i in indices['openai_indexer'] if i != 0]
            assert non_padded_indices == expected_indices[k]
