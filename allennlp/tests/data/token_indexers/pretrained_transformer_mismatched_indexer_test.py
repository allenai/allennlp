from transformers.tokenization_auto import AutoTokenizer

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token, Vocabulary
from allennlp.data.token_indexers import PretrainedTransformerMismatchedIndexer


class TestPretrainedTransformerMismatchedIndexer(AllenNlpTestCase):
    def test_mismatched_behavior(self):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        indexer = PretrainedTransformerMismatchedIndexer("bert-base-cased")
        text = ["AllenNLP", "is", "great"]
        tokens = tokenizer.tokenize(" ".join(["[CLS]"] + text + ["[SEP]"]))
        expected_ids = tokenizer.convert_tokens_to_ids(tokens)
        vocab = Vocabulary()
        indexed = indexer.tokens_to_indices([Token(word) for word in text], vocab)
        assert indexed["token_ids"] == expected_ids
        assert indexed["mask"] == [1] * len(text)
        # Hardcoding a few things because we know how BERT tokenization works
        assert indexed["offsets"] == [(1, 3), (4, 4), (5, 5)]
        assert indexed["wordpiece_mask"] == [1] * len(expected_ids)
