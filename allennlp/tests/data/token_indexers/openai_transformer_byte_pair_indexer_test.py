# pylint: disable=no-self-use,invalid-name
import json
import tarfile

import pytest

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token
from allennlp.data.token_indexers import OpenaiTransformerBytePairIndexer


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

        indices = self.indexer.tokens_to_indices(tokens, None, 'test')

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
            self.indexer.tokens_to_indices(tokens, None, 'should-fail')
