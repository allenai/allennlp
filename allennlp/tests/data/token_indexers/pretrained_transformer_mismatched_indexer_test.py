from transformers.tokenization_auto import AutoTokenizer

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token, Vocabulary
from allennlp.data.token_indexers import PretrainedTransformerMismatchedIndexer


class TestPretrainedTransformerMismatchedIndexer(AllenNlpTestCase):
    def test_bert(self):
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

        keys = indexed.keys()
        assert indexer.get_empty_token_list() == {key: [] for key in keys}

        max_length = 10
        padding_lengths = {key: max_length for key in keys}
        padded_tokens = indexer.as_padded_tensor_dict(indexed, padding_lengths)
        for key in keys:
            padding_length = max_length - len(indexed[key])
            padding = (0, 0) if key == "offsets" else 0
            expected_value = indexed[key] + ([padding] * padding_length)
            assert len(padded_tokens[key]) == max_length
            if key == "offsets":
                expected_value = [list(t) for t in expected_value]
            assert padded_tokens[key].tolist() == expected_value

    def test_auto_determining_num_tokens_added(self):
        indexer = PretrainedTransformerMismatchedIndexer("bert-base-cased")
        assert indexer._determine_num_special_tokens_added() == (1, 1)
