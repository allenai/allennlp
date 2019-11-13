from transformers.tokenization_auto import AutoTokenizer

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token, Vocabulary
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.vocabulary import DEFAULT_SENTENCE_PAIR_SEPARATION_TOKEN


class TestPretrainedTransformerIndexer(AllenNlpTestCase):
    def test_as_array_produces_token_sequence_bert(self):
        # single sentence
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        indexer = PretrainedTransformerIndexer(model_name="bert-base-uncased")
        tokens_no_specials = tokenizer.tokenize("AllenNLP is great")
        tokens_with_specials = ["[CLS]"] + tokens_no_specials + ["[SEP]"]
        expected_ids = tokenizer.convert_tokens_to_ids(tokens_with_specials) # this one does not append any additional tokens
        allennlp_tokens_no_specials = [Token(token) for token in tokens_no_specials]
        vocab = Vocabulary()
        indexed = indexer.tokens_to_indices(allennlp_tokens_no_specials, vocab, "key") # adds special token ids
        assert indexed["key"] == expected_ids

        # sentence pair
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        indexer = PretrainedTransformerIndexer(model_name="bert-base-uncased")
        tokens_1_no_specials = tokenizer.tokenize("AllenNLP is great.")
        tokens_2_no_specials = tokenizer.tokenize("I mean it really is!")
        tokens_with_specials = tokenizer.tokenize("[CLS] AllenNLP is great. [SEP] I mean it really is! [SEP]")

        expected_ids = tokenizer.convert_tokens_to_ids(tokens_with_specials) # this one does not append any additional tokens

        allennlp_tokens_1_no_specials = [Token(token) for token in tokens_1_no_specials]
        allennlp_tokens_2_no_specials = [Token(token) for token in tokens_2_no_specials]
        allennlp_tokens_prepared_no_specials = allennlp_tokens_1_no_specials + [Token(DEFAULT_SENTENCE_PAIR_SEPARATION_TOKEN)] + allennlp_tokens_2_no_specials

        vocab = Vocabulary()
        indexed = indexer.tokens_to_indices(allennlp_tokens_prepared_no_specials, vocab, "key") # adds special token ids
        assert indexed["key"] == expected_ids

    def test_as_array_produces_token_sequence_roberta(self):
        # single sentence
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        indexer = PretrainedTransformerIndexer(model_name="roberta-base")
        tokens_no_specials = tokenizer.tokenize("AllenNLP is great")
        tokens_with_specials = ["<s>"] + tokens_no_specials + ["</s>"]
        expected_ids = tokenizer.convert_tokens_to_ids(tokens_with_specials) # this one does not append any additional tokens
        allennlp_tokens_no_specials = [Token(token) for token in tokens_no_specials]
        vocab = Vocabulary()
        indexed = indexer.tokens_to_indices(allennlp_tokens_no_specials, vocab, "key") # adds special token ids
        assert indexed["key"] == expected_ids

        # sentence pair
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        indexer = PretrainedTransformerIndexer(model_name="roberta-base")
        tokens_1_no_specials = tokenizer.tokenize("AllenNLP is great.")
        tokens_2_no_specials = tokenizer.tokenize("I mean it really is!")
        tokens_with_specials = tokenizer.tokenize("<s> AllenNLP is great. </s> </s> I mean it really is! </s>")

        expected_ids = tokenizer.convert_tokens_to_ids(tokens_with_specials) # this one does not append any additional tokens

        allennlp_tokens_1_no_specials = [Token(token) for token in tokens_1_no_specials]
        allennlp_tokens_2_no_specials = [Token(token) for token in tokens_2_no_specials]
        allennlp_tokens_prepared_no_specials = allennlp_tokens_1_no_specials + [Token(DEFAULT_SENTENCE_PAIR_SEPARATION_TOKEN)] + allennlp_tokens_2_no_specials

        vocab = Vocabulary()
        indexed = indexer.tokens_to_indices(allennlp_tokens_prepared_no_specials, vocab, "key") # adds special token ids
        assert indexed["key"] == expected_ids
