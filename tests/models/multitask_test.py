from allennlp.common.testing import ModelTestCase
from allennlp.data import Instance, Vocabulary, Token
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.models.heads import ClassifierHead
from allennlp.models import MultiTaskModel
from allennlp.modules.backbones import PretrainedTransformerBackbone


class TestMultiTaskModel(ModelTestCase):
    def test_forward_works(self):
        transformer_name = "epwalsh/bert-xsmall-dummy"
        vocab = Vocabulary()
        backbone = PretrainedTransformerBackbone(model_name=transformer_name)
        head = ClassifierHead(vocab, input_dim=20, num_labels=3)
        model = MultiTaskModel(vocab, backbone, {"cls": head})
        tokenizer = PretrainedTransformerTokenizer(model_name=transformer_name)
        token_indexers = PretrainedTransformerIndexer(model_name=transformer_name)
        tokens = tokenizer.tokenize("This is a test")
        text_field = TextField(tokens, {"tokens": token_indexers})
        label_field = LabelField(1, skip_indexing=True)
        instance = Instance({"text": text_field, "label": label_field})
        outputs = model.forward_on_instance(instance)
        assert "loss" in outputs
