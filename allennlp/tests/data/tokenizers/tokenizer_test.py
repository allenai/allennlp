from allennlp.common.testing import AllenNlpTestCase
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from allennlp.data.tokenizers.whitespace_tokenizer import WhitespaceTokenizer


class TestTokenizer(AllenNlpTestCase):
    def test_initializes_from_legacy_word_tokenizer_params(self):
        params = Params(
            {
                "type": "word",
                "word_splitter": {"type": "spacy", "pos_tags": True},
                "start_tokens": ["<s>"],
                "end_tokens": ["</s>"],
            }
        )
        tokenizer = Tokenizer.from_params(params)
        assert isinstance(tokenizer, SpacyTokenizer)
        assert tokenizer._start_tokens == params["start_tokens"]
        assert tokenizer._end_tokens == params["end_tokens"]
        assert "tagger" in tokenizer.spacy.pipe_names

        # Remove "word_splitter_type"
        params = Params({"type": "word", "word_splitter": {"pos_tags": True}})
        tokenizer = Tokenizer.from_params(params)
        assert isinstance(tokenizer, SpacyTokenizer)

        # Splitter is a string
        params = Params({"type": "word", "word_splitter": "just_spaces"})
        tokenizer = Tokenizer.from_params(params)
        assert isinstance(tokenizer, WhitespaceTokenizer)

        # Remove legacy tokenizer type
        params = Params({"word_splitter": "spacy"})
        tokenizer = Tokenizer.from_params(params)
        assert isinstance(tokenizer, SpacyTokenizer)

    def test_raises_exception_for_invalid_legacy_params(self):
        params = Params({"type": "word", "word_stemmer": "porter"})
        with self.assertRaises(ConfigurationError):
            Tokenizer.from_params(params)
        params = Params({"type": "word", "word_filter": "regex"})
        with self.assertRaises(ConfigurationError):
            Tokenizer.from_params(params)
