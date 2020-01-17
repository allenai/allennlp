from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.dataset_readers import NextTokenLmReader
from allennlp.data import Vocabulary
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer


class TestNextTokenLmReader(AllenNlpTestCase):
    def test_text_to_instance_with_basic_tokenizer_and_indexer(self):
        reader = NextTokenLmReader()

        vocab = Vocabulary()
        vocab.add_tokens_to_namespace(["This", "is", "a"], "tokens")

        instance = reader.text_to_instance(sentence="This is a", target="This")
        assert [t.text for t in instance["tokens"]] == ["This", "is", "a"]
        assert [t.text for t in instance["target_ids"]] == ["This"]

        instance.index_fields(vocab)
        tensor_dict = instance.as_tensor_dict(instance.get_padding_lengths())
        assert tensor_dict.keys() == {"tokens", "target_ids"}
        assert tensor_dict["tokens"]["tokens"]["tokens"].numpy().tolist() == [2, 3, 4]
        assert tensor_dict["target_ids"]["tokens"]["tokens"].numpy().tolist() == [2]

    def test_text_to_instance_with_bert_tokenizer_and_indexer(self):
        tokenizer = PretrainedTransformerTokenizer("bert-base-cased")
        token_indexer = PretrainedTransformerIndexer("bert-base-cased")
        reader = NextTokenLmReader(tokenizer, {"bert": token_indexer})
        instance = reader.text_to_instance(sentence="AllenNLP is very", target="very")
        assert [t.text for t in instance["tokens"]] == [
            "[CLS]",
            "Allen",
            "##NL",
            "##P",
            "is",
            "very",
            "[SEP]",
        ]
        assert [t.text for t in instance["target_ids"]] == ["very"]

        vocab = Vocabulary()
        instance.index_fields(vocab)
        tensor_dict = instance.as_tensor_dict(instance.get_padding_lengths())
        assert tensor_dict.keys() == {"tokens", "target_ids"}
        bert_token_ids = tensor_dict["tokens"]["bert"]["token_ids"].numpy().tolist()
        target_ids = tensor_dict["target_ids"]["bert"]["token_ids"].numpy().tolist()
        # I don't know what wordpiece id BERT is going to assign to 'This', but it at least should
        # be the same between the input and the target.
        assert target_ids[0] == bert_token_ids[5]
