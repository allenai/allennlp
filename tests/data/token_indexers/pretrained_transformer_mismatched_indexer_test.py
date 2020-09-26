from allennlp.common import cached_transformers
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token, Vocabulary
from allennlp.data.token_indexers import PretrainedTransformerMismatchedIndexer


class TestPretrainedTransformerMismatchedIndexer(AllenNlpTestCase):
    def test_bert(self):
        tokenizer = cached_transformers.get_tokenizer("bert-base-cased")
        indexer = PretrainedTransformerMismatchedIndexer("bert-base-cased")
        text = ["AllenNLP", "is", "great"]
        tokens = tokenizer.tokenize(" ".join(["[CLS]"] + text + ["[SEP]"]))
        expected_ids = tokenizer.convert_tokens_to_ids(tokens)
        vocab = Vocabulary()
        indexed = indexer.tokens_to_indices([Token(word) for word in text], vocab)
        assert indexed["token_ids"] == expected_ids
        assert indexed["mask"] == [True] * len(text)
        # Hardcoding a few things because we know how BERT tokenization works
        assert indexed["offsets"] == [(1, 3), (4, 4), (5, 5)]
        assert indexed["wordpiece_mask"] == [True] * len(expected_ids)

        keys = indexed.keys()
        assert indexer.get_empty_token_list() == {key: [] for key in keys}

        max_length = 10
        padding_lengths = {key: max_length for key in keys}
        padded_tokens = indexer.as_padded_tensor_dict(indexed, padding_lengths)
        for key in keys:
            padding_length = max_length - len(indexed[key])
            if key == "offsets":
                padding = (0, 0)
            elif "mask" in key:
                padding = False
            else:
                padding = 0
            expected_value = indexed[key] + ([padding] * padding_length)
            assert len(padded_tokens[key]) == max_length
            if key == "offsets":
                expected_value = [list(t) for t in expected_value]
            assert padded_tokens[key].tolist() == expected_value

    def test_long_sequence_splitting(self):
        tokenizer = cached_transformers.get_tokenizer("bert-base-uncased")
        indexer = PretrainedTransformerMismatchedIndexer("bert-base-uncased", max_length=4)
        text = ["AllenNLP", "is", "great"]
        tokens = tokenizer.tokenize(" ".join(["[CLS]"] + text + ["[SEP]"]))
        expected_ids = tokenizer.convert_tokens_to_ids(tokens)
        assert len(expected_ids) == 7  # just to make sure it's what we're expecting
        cls_id, sep_id = expected_ids[0], expected_ids[-1]
        expected_ids = (
            expected_ids[:3]
            + [sep_id, cls_id]
            + expected_ids[3:5]
            + [sep_id, cls_id]
            + expected_ids[5:]
        )

        vocab = Vocabulary()
        indexed = indexer.tokens_to_indices([Token(word) for word in text], vocab)

        assert indexed["token_ids"] == expected_ids
        # [CLS] allen ##nl [SEP] [CLS] #p is [SEP] [CLS] great [SEP]
        assert indexed["segment_concat_mask"] == [True] * len(expected_ids)
        # allennlp is great
        assert indexed["mask"] == [True] * len(text)
        # [CLS] allen #nl #p is great [SEP]
        assert indexed["wordpiece_mask"] == [True] * 7
