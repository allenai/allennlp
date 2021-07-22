import torch
import pytest

from allennlp.common.cached_transformers import get_tokenizer
from allennlp.data import Batch, Instance
from allennlp.data.fields import TransformerTextField


def test_transformer_text_field_init():
    field = TransformerTextField(torch.IntTensor([1, 2, 3]))
    field_as_tensor = field.as_tensor(field.get_padding_lengths())
    assert "input_ids" in field_as_tensor
    assert "attention_mask" in field_as_tensor
    assert torch.all(field_as_tensor["attention_mask"] == torch.BoolTensor([True, True, True]))
    assert torch.all(field_as_tensor["input_ids"] == torch.IntTensor([1, 2, 3]))


def test_empty_transformer_text_field():
    field = TransformerTextField(torch.IntTensor([]), padding_token_id=9)
    field = field.empty_field()
    assert isinstance(field, TransformerTextField) and field.padding_token_id == 9
    field_as_tensor = field.as_tensor(field.get_padding_lengths())
    assert "input_ids" in field_as_tensor
    assert "attention_mask" in field_as_tensor
    assert torch.all(field_as_tensor["attention_mask"] == torch.BoolTensor([]))
    assert torch.all(field_as_tensor["input_ids"] == torch.IntTensor([]))


def test_transformer_text_field_batching():
    batch = Batch(
        [
            Instance({"text": TransformerTextField(torch.IntTensor([1, 2, 3]))}),
            Instance({"text": TransformerTextField(torch.IntTensor([2, 3, 4, 5]))}),
            Instance({"text": TransformerTextField(torch.IntTensor())}),
        ]
    )
    tensors = batch.as_tensor_dict(batch.get_padding_lengths())
    assert tensors["text"]["input_ids"].shape == (3, 4)
    assert tensors["text"]["input_ids"][0, -1] == 0
    assert tensors["text"]["attention_mask"][0, -1] == torch.Tensor([False])
    assert torch.all(tensors["text"]["input_ids"][-1] == 0)
    assert torch.all(tensors["text"]["attention_mask"][-1] == torch.tensor([False]))


@pytest.mark.parametrize("return_tensors", ["pt", None])
def test_transformer_text_field_from_huggingface(return_tensors):
    tokenizer = get_tokenizer("bert-base-cased")

    batch = Batch(
        [
            Instance(
                {"text": TransformerTextField(**tokenizer(text, return_tensors=return_tensors))}
            )
            for text in [
                "Hello, World!",
                "The fox jumped over the fence",
                "Humpty dumpty sat on a wall",
            ]
        ]
    )
    tensors = batch.as_tensor_dict(batch.get_padding_lengths())
    assert tensors["text"]["input_ids"].shape == (3, 11)
