from transformers.tokenization_auto import AutoTokenizer

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token, Vocabulary
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer


class TestPretrainedTransformerIndexer(AllenNlpTestCase):
    def test_as_array_produces_token_sequence_bert(self):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        allennlp_tokenier = PretrainedTransformerTokenizer("bert-base-uncased")

        # uncased
        indexer = PretrainedTransformerIndexer(model_name="bert-base-uncased")
        string_specials = "[CLS] AllenNLP is great [SEP]"
        string_no_specials = "AllenNLP is great"
        tokens = tokenizer.tokenize(string_specials)
        expected_ids = tokenizer.convert_tokens_to_ids(tokens)
        # tokens tokenized with our pretrained tokenizer have indices in them
        allennlp_tokens = allennlp_tokenier.tokenize(string_no_specials)
        vocab = Vocabulary()
        indexed = indexer.tokens_to_indices(allennlp_tokens, vocab, "key")
        assert indexed["key"] == expected_ids

        # cased
        indexer = PretrainedTransformerIndexer(model_name="bert-base-cased")
        string_specials = "[CLS] AllenNLP is great [SEP]"
        string_no_specials = "AllenNLP is great"
        tokens = tokenizer.tokenize(string_specials)
        expected_ids = tokenizer.convert_tokens_to_ids(tokens)
        # tokens tokenized with our pretrained tokenizer have indices in them
        allennlp_tokens = allennlp_tokenier.tokenize(string_no_specials)
        vocab = Vocabulary()
        indexed = indexer.tokens_to_indices(allennlp_tokens, vocab, "key")
        assert indexed["key"] == expected_ids

        # sentence pair
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        indexer = PretrainedTransformerIndexer(model_name="bert-base-uncased")
        default_format = "[CLS] AllenNLP is great! [SEP] Really it is! [SEP]"
        tokens = tokenizer.tokenize(default_format)
        expected_ids = tokenizer.convert_tokens_to_ids(tokens)
        allennlp_tokens = allennlp_tokenier.tokenize_sentences(
            ["AllenNLP is great!", "Really it is!"]
        )
        vocab = Vocabulary()
        indexed = indexer.tokens_to_indices(allennlp_tokens, vocab, "key")
        assert indexed["key"] == expected_ids

    def test_as_array_produces_token_sequence_roberta(self):
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        allennlp_tokenier = PretrainedTransformerTokenizer("roberta-base")

        # single
        indexer = PretrainedTransformerIndexer(model_name="roberta-base")
        string_specials = "<s> AllenNLP is great </s>"
        string_no_specials = "AllenNLP is great"
        tokens = tokenizer.tokenize(string_specials)
        expected_ids = tokenizer.convert_tokens_to_ids(tokens)
        # tokens tokenized with our pretrained tokenizer have indices in them
        allennlp_tokens = allennlp_tokenier.tokenize(string_no_specials)
        vocab = Vocabulary()
        indexed = indexer.tokens_to_indices(allennlp_tokens, vocab, "key")
        assert indexed["key"] == expected_ids

        # pair
        default_format = "<s> AllenNLP is great! </s> </s> Really it is! </s>"
        tokens = tokenizer.tokenize(default_format)
        expected_ids = tokenizer.convert_tokens_to_ids(tokens)
        allennlp_tokens = allennlp_tokenier.tokenize_sentences(
            ["AllenNLP is great!", "Really it is!"]
        )
        vocab = Vocabulary()
        indexed = indexer.tokens_to_indices(allennlp_tokens, vocab, "key")
        assert indexed["key"] == expected_ids
