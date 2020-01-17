from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.dataset_readers import MaskedLanguageModelingReader
from allennlp.data import Vocabulary
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer


class TestMaskedLanguageModelingDatasetReader(AllenNlpTestCase):
    def test_text_to_instance_with_basic_tokenizer_and_indexer(self):
        reader = MaskedLanguageModelingReader()

        vocab = Vocabulary()
        vocab.add_tokens_to_namespace(["This", "is", "a", "[MASK]", "token", "."], "tokens")

        instance = reader.text_to_instance(sentence="This is a [MASK] token .", targets=["This"])
        assert [t.text for t in instance["tokens"]] == ["This", "is", "a", "[MASK]", "token", "."]
        assert [i.sequence_index for i in instance["mask_positions"]] == [3]
        assert [t.text for t in instance["target_ids"]] == ["This"]

        instance.index_fields(vocab)
        tensor_dict = instance.as_tensor_dict(instance.get_padding_lengths())
        assert tensor_dict.keys() == {"tokens", "mask_positions", "target_ids"}
        assert tensor_dict["tokens"]["tokens"]["tokens"].numpy().tolist() == [2, 3, 4, 5, 6, 7]
        assert tensor_dict["target_ids"]["tokens"]["tokens"].numpy().tolist() == [2]
        assert tensor_dict["mask_positions"].numpy().tolist() == [[3]]

    def test_text_to_instance_with_bert_tokenizer_and_indexer(self):
        tokenizer = PretrainedTransformerTokenizer("bert-base-cased")
        token_indexer = PretrainedTransformerIndexer("bert-base-cased")
        reader = MaskedLanguageModelingReader(tokenizer, {"bert": token_indexer})
        instance = reader.text_to_instance(
            sentence="This is AllenNLP [MASK] token .", targets=["This"]
        )
        assert [t.text for t in instance["tokens"]] == [
            "[CLS]",
            "This",
            "is",
            "Allen",
            "##NL",
            "##P",
            "[MASK]",
            "token",
            ".",
            "[SEP]",
        ]
        assert [i.sequence_index for i in instance["mask_positions"]] == [6]
        assert [t.text for t in instance["target_ids"]] == ["This"]

        vocab = Vocabulary()
        instance.index_fields(vocab)
        tensor_dict = instance.as_tensor_dict(instance.get_padding_lengths())
        assert tensor_dict.keys() == {"tokens", "mask_positions", "target_ids"}
        bert_token_ids = tensor_dict["tokens"]["bert"]["token_ids"].numpy().tolist()
        target_ids = tensor_dict["target_ids"]["bert"]["token_ids"].numpy().tolist()
        # I don't know what wordpiece id BERT is going to assign to 'This', but it at least should
        # be the same between the input and the target.
        assert target_ids[0] == bert_token_ids[1]
