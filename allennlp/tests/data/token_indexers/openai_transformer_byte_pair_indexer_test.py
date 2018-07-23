# pylint: disable=no-self-use,invalid-name
import json

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token
from allennlp.data.token_indexers import OpenaiTransformerBytePairIndexer


class TestOpenaiTransformerBytePairIndexer(AllenNlpTestCase):
    def setUp(self):
        super().setUp()

        self.encoder_path = self.TEST_DIR / 'encoder.json'
        self.bpe_path = self.TEST_DIR / 'vocab.bpe'

        symbols = ["e", "w", "o", "wo", "."]
        byte_pairs = [(sym1, sym2 + end)
                      for sym1 in symbols        # prefer earlier first symbol
                      for sym2 in symbols        # if tie, prefer earlier second symbol
                      for end in ('</w>', '')]   # if tie, prefer ending a word
        encoding = {f"{sym1}{sym2}": idx + 1 for idx, (sym1, sym2) in enumerate(byte_pairs)}

        with open(self.encoder_path, 'w') as encoder_file:
            json.dump(encoding, encoder_file)

        with open(self.bpe_path, 'w') as bpe_file:
            bpe_file.write("#version 0.0\n")
            for sym1, sym2 in byte_pairs:
                bpe_file.write(f"{sym1} {sym2}\n")
            bpe_file.write("\n")

        self.indexer = OpenaiTransformerBytePairIndexer(self.encoder_path, self.bpe_path)

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

        assert set(indices.keys()) == {"test-text_tokens", "test-offsets"}

        text_tokens = indices['test-text_tokens']
        offsets = indices['test-offsets']

        assert text_tokens == [
                self.indexer.encoder.get(symbol, 0)
                for symbol in ['ew', 'oe</w>'] + ['woe</w>'] + ['ew', 'e</w>'] + ['ee</w>']
        ]

        assert offsets == [
                1,  # end of first word
                2,  # end of second word
                4,  # end of third word
                5,  # end of last word
        ]
