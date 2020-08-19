from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token


class TestInstance(AllenNlpTestCase):
    def test_instance_implements_mutable_mapping(self):
        words_field = TextField([Token("hello")], {})
        label_field = LabelField(1, skip_indexing=True)
        instance = Instance({"words": words_field, "labels": label_field})

        assert instance["words"] == words_field
        assert instance["labels"] == label_field
        assert len(instance) == 2

        keys = {k for k, v in instance.items()}
        assert keys == {"words", "labels"}

        values = [v for k, v in instance.items()]
        assert words_field in values
        assert label_field in values

    def test_duplicate(self):
        # Verify the `duplicate()` method works with a `PretrainedTransformerIndexer` in
        # a `TextField`. See https://github.com/allenai/allennlp/issues/4270.
        instance = Instance(
            {
                "words": TextField(
                    [Token("hello")], {"tokens": PretrainedTransformerIndexer("bert-base-uncased")}
                )
            }
        )

        other = instance.duplicate()
        assert other == instance

        # Adding new fields to the original instance should not effect the duplicate.
        instance.add_field("labels", LabelField("some_label"))
        assert "labels" not in other.fields
        assert other != instance  # sanity check on the '__eq__' method.
