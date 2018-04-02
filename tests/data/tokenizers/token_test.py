# pylint: disable=no-self-use
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import get_spacy_model
from allennlp.data.tokenizers.token import Token, truncate_token, token_to_json, json_to_token

class TestTokens(AllenNlpTestCase):
    def test_truncate_spacy_tokens(self):
        spacy = get_spacy_model('en_core_web_sm', False, False, False)
        tokens = [t for t in spacy("fight the establishment forever")]
        assert len(tokens) == 4
        assert tokens[2].text == "establishment"

        truncated = [truncate_token(token, 9) for token in tokens]

        for i in range(4):
            assert truncated[i].text == tokens[i].text[:9]
            assert truncated[i].idx == tokens[i].idx

    def test_truncate_allennlp_tokens(self):
        token = Token('establishment', idx=10)
        truncated = truncate_token(token, 9)
        assert truncated.text == "establish"
        assert truncated.idx == 10

    def test_token_to_json_short(self):
        spacy = get_spacy_model('en_core_web_sm', False, False, False)
        tokens = [t for t in spacy("fight the establishment forever")]
        jsons = [token_to_json(token, short=True) for token in tokens]
        assert jsons == [
                ['fight', 0],
                ['the', 6],
                ['establishment', 10],
                ['forever', 24]
        ]

        tokens2 = [json_to_token(blob, short=True, max_len=9) for blob in jsons]
        assert tokens2[2].text == "establish"
        assert tokens2[3].text == "forever"
        assert tokens2[3].idx == 24

    def test_token_to_json_long(self):
        spacy = get_spacy_model('en_core_web_sm', False, False, False)
        tokens = [t for t in spacy("fight the establishment forever")]
        jsons = [token_to_json(token, short=False) for token in tokens]
        assert jsons[2]["text"] == "establishment"
        assert jsons[2]["idx"] == 10

        tokens2 = [json_to_token(blob, short=False, max_len=9) for blob in jsons]
        assert tokens2[2].text == "establish"
        assert tokens2[3].text == "forever"
        assert tokens2[3].idx == 24
