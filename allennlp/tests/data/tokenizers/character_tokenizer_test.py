from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers import CharacterTokenizer


class TestCharacterTokenizer(AllenNlpTestCase):
    def test_splits_into_characters(self):
        tokenizer = CharacterTokenizer(start_tokens=["<S1>", "<S2>"], end_tokens=["</S2>", "</S1>"])
        sentence = "A, small sentence."
        tokens = [t.text for t in tokenizer.tokenize(sentence)]
        expected_tokens = [
            "<S1>",
            "<S2>",
            "A",
            ",",
            " ",
            "s",
            "m",
            "a",
            "l",
            "l",
            " ",
            "s",
            "e",
            "n",
            "t",
            "e",
            "n",
            "c",
            "e",
            ".",
            "</S2>",
            "</S1>",
        ]
        assert tokens == expected_tokens

    def test_batch_tokenization(self):
        tokenizer = CharacterTokenizer()
        sentences = [
            "This is a sentence",
            "This isn't a sentence.",
            "This is the 3rd sentence." "Here's the 'fourth' sentence.",
        ]
        batch_tokenized = tokenizer.batch_tokenize(sentences)
        separately_tokenized = [tokenizer.tokenize(sentence) for sentence in sentences]
        assert len(batch_tokenized) == len(separately_tokenized)
        for batch_sentence, separate_sentence in zip(batch_tokenized, separately_tokenized):
            assert len(batch_sentence) == len(separate_sentence)
            for batch_word, separate_word in zip(batch_sentence, separate_sentence):
                assert batch_word.text == separate_word.text

    def test_handles_byte_encoding(self):
        tokenizer = CharacterTokenizer(byte_encoding="utf-8", start_tokens=[259], end_tokens=[260])
        word = "åøâáabe"
        tokens = [t.text_id for t in tokenizer.tokenize(word)]
        # Note that we've added one to the utf-8 encoded bytes, to account for masking.
        expected_tokens = [259, 196, 166, 196, 185, 196, 163, 196, 162, 98, 99, 102, 260]
        assert tokens == expected_tokens
