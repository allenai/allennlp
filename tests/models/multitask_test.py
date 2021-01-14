import pytest

from allennlp.common.testing import ModelTestCase
from allennlp.data import Instance, Vocabulary, Batch
from allennlp.data.fields import LabelField, TextField, MetadataField
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.models.heads import ClassifierHead
from allennlp.models import MultiTaskModel
from allennlp.modules.backbones import PretrainedTransformerBackbone
from allennlp.modules.seq2vec_encoders import ClsPooler


class TestMultiTaskModel(ModelTestCase):
    def test_forward_works(self):
        # Setting up the model.
        transformer_name = "epwalsh/bert-xsmall-dummy"
        vocab = Vocabulary()
        backbone = PretrainedTransformerBackbone(vocab, transformer_name)
        head1 = ClassifierHead(vocab, seq2vec_encoder=ClsPooler(20), input_dim=20, num_labels=3)
        head2 = ClassifierHead(vocab, seq2vec_encoder=ClsPooler(20), input_dim=20, num_labels=4)
        # We'll start with one head, and add another later.
        model = MultiTaskModel(vocab, backbone, {"cls": head1})

        # Setting up the data.
        tokenizer = PretrainedTransformerTokenizer(model_name=transformer_name)
        token_indexers = PretrainedTransformerIndexer(model_name=transformer_name)
        tokens = tokenizer.tokenize("This is a test")
        text_field = TextField(tokens, {"tokens": token_indexers})
        label_field1 = LabelField(1, skip_indexing=True)
        label_field2 = LabelField(3, skip_indexing=True)
        instance = Instance(
            {"text": text_field, "label": label_field1, "task": MetadataField("cls")}
        )

        # Now we run some tests.  First, the default.
        outputs = model.forward_on_instance(instance)
        assert "encoded_text" in outputs
        assert "cls_logits" in outputs
        assert "loss" in outputs
        assert "cls_loss" in outputs

        # When we don't have labels.
        instance = Instance({"text": text_field, "task": MetadataField("cls")})
        outputs = model.forward_on_instance(instance)
        assert "encoded_text" in outputs
        assert "cls_logits" in outputs
        assert "loss" not in outputs

        # Same in eval mode
        model.eval()
        outputs = model.forward_on_instance(instance)
        assert "encoded_text" in outputs
        assert "loss" not in outputs  # no loss because we have no labels
        assert "cls_logits" in outputs  # but we can compute logits
        model.train()

        # Now for two headed and other more complex tests.
        model = MultiTaskModel(
            vocab,
            backbone,
            {"cls1": head1, "cls2": head2},
            arg_name_mapping={
                "backbone": {"question": "text"},
            },
        )

        # Basic case where things should work, with two heads that both need label inputs.
        instance1 = Instance(
            {"text": text_field, "label": label_field1, "task": MetadataField("cls1")}
        )
        instance2 = Instance(
            {"text": text_field, "label": label_field2, "task": MetadataField("cls2")}
        )
        batch = Batch([instance1, instance2])
        outputs = model.forward(**batch.as_tensor_dict())
        assert "encoded_text" in outputs
        assert "cls1_logits" in outputs
        assert "cls1_loss" in outputs
        assert "cls2_logits" in outputs
        assert "cls2_loss" in outputs
        assert "loss" in outputs
        combined_loss = outputs["cls1_loss"].item() + outputs["cls2_loss"].item()
        assert abs(outputs["loss"].item() - combined_loss) <= 1e-6

        # This should fail, because we're using task 'cls1' with the labels for `cls2`, and the sizes don't match.
        # This shows up as an IndexError in this case. It'd be nice to catch this kind of error more cleanly in the
        # model class, but I'm not sure how.
        instance = Instance(
            {"text": text_field, "label": label_field2, "task": MetadataField("cls1")}
        )
        with pytest.raises(IndexError):
            outputs = model.forward_on_instance(instance)

        # This one should fail because we now have two things that map to "text" in the backbone,
        # and they would clobber each other. The name mapping that we have in the model is ok, as
        # long as our data loader is set up such that we don't batch instances that have both of
        # these fields at the same time.
        instance = Instance(
            {"question": text_field, "text": text_field, "task": MetadataField("cls1")}
        )
        with pytest.raises(ValueError, match="duplicate argument text"):
            outputs = model.forward_on_instance(instance)
