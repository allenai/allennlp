from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers.pretrained_transformer_pre_tokenizer import (
    OpenAIPreTokenizer,
    BertPreTokenizer,
)


class TestOpenAiPreTokenizer(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.word_tokenizer = OpenAIPreTokenizer()

    def test_tokenize_handles_complex_punctuation(self):
        sentence = "This sentence ?a!?!"
        expected_tokens = ["This", "sentence", "?", "a", "!", "?", "!"]
        tokens = [t.text for t in self.word_tokenizer.tokenize(sentence)]
        assert tokens == expected_tokens


class TestBertPreTokenizer(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.word_tokenizer = BertPreTokenizer()

    def test_never_split(self):
        sentence = "[unused0] [UNK] [SEP] [PAD] [CLS] [MASK]"
        expected_tokens = ["[", "unused0", "]", "[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"]
        tokens = [token.text for token in self.word_tokenizer.tokenize(sentence)]
        assert tokens == expected_tokens

    def test_do_lower_case(self):
        # BertPreTokenizer makes every token not in `never_split` to lowercase by default
        word_tokenizer = BertPreTokenizer(never_split=["[UNUSED0]"])
        sentence = "[UNUSED0] [UNK] [unused0]"
        expected_tokens = ["[UNUSED0]", "[", "unk", "]", "[", "unused0", "]"]
        tokens = [token.text for token in word_tokenizer.tokenize(sentence)]
        assert tokens == expected_tokens
