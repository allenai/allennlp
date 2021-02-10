from typing import Dict

import numpy
import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token, Vocabulary, Instance
from allennlp.data.fields import TextField, LabelField, ListField, IndexField, SequenceLabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.dataloader import PyTorchDataLoader
from allennlp.data.dataset_readers.dataset_reader import AllennlpDataset
from allennlp.data.tokenizers import SpacyTokenizer
from allennlp.models import Model
from allennlp.modules import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder


class DummyModel(Model):
    """
    Performs a common operation (embedding) that won't work on an empty tensor.
    Returns an arbitrary loss.
    """

    def __init__(self, vocab: Vocabulary) -> None:
        super().__init__(vocab)
        weight = torch.ones(vocab.get_vocab_size(), 10)
        token_embedding = Embedding(
            num_embeddings=vocab.get_vocab_size(), embedding_dim=10, weight=weight, trainable=False
        )
        self.embedder = BasicTextFieldEmbedder({"words": token_embedding})

    def forward(  # type: ignore
        self, list_tensor: Dict[str, torch.LongTensor]
    ) -> Dict[str, torch.Tensor]:
        self.embedder(list_tensor)
        return {"loss": 1.0}


class TestListField(AllenNlpTestCase):
    def setup_method(self):
        self.vocab = Vocabulary()
        self.vocab.add_token_to_namespace("this", "words")
        self.vocab.add_token_to_namespace("is", "words")
        self.vocab.add_token_to_namespace("a", "words")
        self.vocab.add_token_to_namespace("sentence", "words")
        self.vocab.add_token_to_namespace("s", "characters")
        self.vocab.add_token_to_namespace("e", "characters")
        self.vocab.add_token_to_namespace("n", "characters")
        self.vocab.add_token_to_namespace("t", "characters")
        self.vocab.add_token_to_namespace("c", "characters")
        for label in ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]:
            self.vocab.add_token_to_namespace(label, "labels")

        self.word_indexer = {"words": SingleIdTokenIndexer("words")}
        self.words_and_characters_indexers = {
            "words": SingleIdTokenIndexer("words"),
            "characters": TokenCharactersIndexer("characters", min_padding_length=1),
        }
        self.field1 = TextField(
            [Token(t) for t in ["this", "is", "a", "sentence"]], self.word_indexer
        )
        self.field2 = TextField(
            [Token(t) for t in ["this", "is", "a", "different", "sentence"]], self.word_indexer
        )
        self.field3 = TextField(
            [Token(t) for t in ["this", "is", "another", "sentence"]], self.word_indexer
        )

        self.empty_text_field = self.field1.empty_field()
        self.index_field = IndexField(1, self.field1)
        self.empty_index_field = self.index_field.empty_field()
        self.sequence_label_field = SequenceLabelField([1, 1, 0, 1], self.field1)
        self.empty_sequence_label_field = self.sequence_label_field.empty_field()

        tokenizer = SpacyTokenizer()
        tokens = tokenizer.tokenize("Foo")
        text_field = TextField(tokens, self.word_indexer)
        empty_list_field = ListField([text_field.empty_field()])
        empty_fields = {"list_tensor": empty_list_field}
        self.empty_instance = Instance(empty_fields)

        non_empty_list_field = ListField([text_field])
        non_empty_fields = {"list_tensor": non_empty_list_field}
        self.non_empty_instance = Instance(non_empty_fields)

        super().setup_method()

    def test_get_padding_lengths(self):
        list_field = ListField([self.field1, self.field2, self.field3])
        list_field.index(self.vocab)
        lengths = list_field.get_padding_lengths()
        assert lengths == {"num_fields": 3, "list_words___tokens": 5}

    def test_list_field_can_handle_empty_text_fields(self):
        list_field = ListField([self.field1, self.field2, self.empty_text_field])
        list_field.index(self.vocab)
        tensor_dict = list_field.as_tensor(list_field.get_padding_lengths())
        numpy.testing.assert_array_equal(
            tensor_dict["words"]["tokens"].detach().cpu().numpy(),
            numpy.array([[2, 3, 4, 5, 0], [2, 3, 4, 1, 5], [0, 0, 0, 0, 0]]),
        )

    def test_list_field_can_handle_empty_index_fields(self):
        list_field = ListField([self.index_field, self.index_field, self.empty_index_field])
        list_field.index(self.vocab)
        tensor = list_field.as_tensor(list_field.get_padding_lengths())
        numpy.testing.assert_array_equal(
            tensor.detach().cpu().numpy(), numpy.array([[1], [1], [-1]])
        )

    def test_list_field_can_handle_empty_sequence_label_fields(self):
        list_field = ListField(
            [self.sequence_label_field, self.sequence_label_field, self.empty_sequence_label_field]
        )
        list_field.index(self.vocab)
        tensor = list_field.as_tensor(list_field.get_padding_lengths())
        numpy.testing.assert_array_equal(
            tensor.detach().cpu().numpy(), numpy.array([[1, 1, 0, 1], [1, 1, 0, 1], [0, 0, 0, 0]])
        )

    def test_all_fields_padded_to_max_length(self):
        list_field = ListField([self.field1, self.field2, self.field3])
        list_field.index(self.vocab)
        tensor_dict = list_field.as_tensor(list_field.get_padding_lengths())
        numpy.testing.assert_array_almost_equal(
            tensor_dict["words"]["tokens"][0].detach().cpu().numpy(), numpy.array([2, 3, 4, 5, 0])
        )
        numpy.testing.assert_array_almost_equal(
            tensor_dict["words"]["tokens"][1].detach().cpu().numpy(), numpy.array([2, 3, 4, 1, 5])
        )
        numpy.testing.assert_array_almost_equal(
            tensor_dict["words"]["tokens"][2].detach().cpu().numpy(), numpy.array([2, 3, 1, 5, 0])
        )

    def test_nested_list_fields_are_padded_correctly(self):
        nested_field1 = ListField([LabelField(c) for c in ["a", "b", "c", "d", "e"]])
        nested_field2 = ListField([LabelField(c) for c in ["f", "g", "h", "i", "j", "k"]])
        list_field = ListField([nested_field1.empty_field(), nested_field1, nested_field2])
        list_field.index(self.vocab)
        padding_lengths = list_field.get_padding_lengths()
        assert padding_lengths == {"num_fields": 3, "list_num_fields": 6}
        tensor = list_field.as_tensor(padding_lengths).detach().cpu().numpy()
        numpy.testing.assert_almost_equal(
            tensor, [[-1, -1, -1, -1, -1, -1], [0, 1, 2, 3, 4, -1], [5, 6, 7, 8, 9, 10]]
        )

    def test_fields_can_pad_to_greater_than_max_length(self):
        list_field = ListField([self.field1, self.field2, self.field3])
        list_field.index(self.vocab)
        padding_lengths = list_field.get_padding_lengths()
        padding_lengths["list_words___tokens"] = 7
        padding_lengths["num_fields"] = 5
        tensor_dict = list_field.as_tensor(padding_lengths)
        numpy.testing.assert_array_almost_equal(
            tensor_dict["words"]["tokens"][0].detach().cpu().numpy(),
            numpy.array([2, 3, 4, 5, 0, 0, 0]),
        )
        numpy.testing.assert_array_almost_equal(
            tensor_dict["words"]["tokens"][1].detach().cpu().numpy(),
            numpy.array([2, 3, 4, 1, 5, 0, 0]),
        )
        numpy.testing.assert_array_almost_equal(
            tensor_dict["words"]["tokens"][2].detach().cpu().numpy(),
            numpy.array([2, 3, 1, 5, 0, 0, 0]),
        )
        numpy.testing.assert_array_almost_equal(
            tensor_dict["words"]["tokens"][3].detach().cpu().numpy(),
            numpy.array([0, 0, 0, 0, 0, 0, 0]),
        )
        numpy.testing.assert_array_almost_equal(
            tensor_dict["words"]["tokens"][4].detach().cpu().numpy(),
            numpy.array([0, 0, 0, 0, 0, 0, 0]),
        )

    def test_as_tensor_can_handle_multiple_token_indexers(self):

        self.field1._token_indexers = self.words_and_characters_indexers
        self.field2._token_indexers = self.words_and_characters_indexers
        self.field3._token_indexers = self.words_and_characters_indexers

        list_field = ListField([self.field1, self.field2, self.field3])
        list_field.index(self.vocab)
        padding_lengths = list_field.get_padding_lengths()
        tensor_dict = list_field.as_tensor(padding_lengths)
        words = tensor_dict["words"]["tokens"].detach().cpu().numpy()
        characters = tensor_dict["characters"]["token_characters"].detach().cpu().numpy()
        numpy.testing.assert_array_almost_equal(
            words, numpy.array([[2, 3, 4, 5, 0], [2, 3, 4, 1, 5], [2, 3, 1, 5, 0]])
        )

        numpy.testing.assert_array_almost_equal(
            characters[0],
            numpy.array(
                [
                    [5, 1, 1, 2, 0, 0, 0, 0, 0],
                    [1, 2, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [2, 3, 4, 5, 3, 4, 6, 3, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
        )

        numpy.testing.assert_array_almost_equal(
            characters[1],
            numpy.array(
                [
                    [5, 1, 1, 2, 0, 0, 0, 0, 0],
                    [1, 2, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 3, 1, 3, 4, 5],
                    [2, 3, 4, 5, 3, 4, 6, 3, 0],
                ]
            ),
        )

        numpy.testing.assert_array_almost_equal(
            characters[2],
            numpy.array(
                [
                    [5, 1, 1, 2, 0, 0, 0, 0, 0],
                    [1, 2, 0, 0, 0, 0, 0, 0, 0],
                    [1, 4, 1, 5, 1, 3, 1, 0, 0],
                    [2, 3, 4, 5, 3, 4, 6, 3, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
        )

    def test_as_tensor_can_handle_multiple_token_indexers_and_empty_fields(self):

        self.field1._token_indexers = self.words_and_characters_indexers
        self.field2._token_indexers = self.words_and_characters_indexers
        self.field3._token_indexers = self.words_and_characters_indexers

        list_field = ListField([self.field1.empty_field(), self.field1, self.field2])
        list_field.index(self.vocab)
        padding_lengths = list_field.get_padding_lengths()
        tensor_dict = list_field.as_tensor(padding_lengths)
        words = tensor_dict["words"]["tokens"].detach().cpu().numpy()
        characters = tensor_dict["characters"]["token_characters"].detach().cpu().numpy()

        numpy.testing.assert_array_almost_equal(
            words, numpy.array([[0, 0, 0, 0, 0], [2, 3, 4, 5, 0], [2, 3, 4, 1, 5]])
        )

        numpy.testing.assert_array_almost_equal(characters[0], numpy.zeros([5, 9]))

        numpy.testing.assert_array_almost_equal(
            characters[1],
            numpy.array(
                [
                    [5, 1, 1, 2, 0, 0, 0, 0, 0],
                    [1, 2, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [2, 3, 4, 5, 3, 4, 6, 3, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
        )

        numpy.testing.assert_array_almost_equal(
            characters[2],
            numpy.array(
                [
                    [5, 1, 1, 2, 0, 0, 0, 0, 0],
                    [1, 2, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 3, 1, 3, 4, 5],
                    [2, 3, 4, 5, 3, 4, 6, 3, 0],
                ]
            ),
        )

    def test_printing_doesnt_crash(self):
        list_field = ListField([self.field1, self.field2])
        print(list_field)

    def test_sequence_methods(self):
        list_field = ListField([self.field1, self.field2, self.field3])

        assert len(list_field) == 3
        assert list_field[1] == self.field2
        assert [f for f in list_field] == [self.field1, self.field2, self.field3]

    def test_empty_list_can_be_tensorized(self):
        tokenizer = SpacyTokenizer()
        tokens = tokenizer.tokenize("Foo")
        text_field = TextField(tokens, self.word_indexer)
        list_field = ListField([text_field.empty_field()])
        fields = {
            "list": list_field,
            "bar": TextField(tokenizer.tokenize("BAR"), self.word_indexer),
        }
        instance = Instance(fields)
        instance.index_fields(self.vocab)
        instance.as_tensor_dict()

    def test_batch_with_some_empty_lists_works(self):
        dataset = AllennlpDataset([self.empty_instance, self.non_empty_instance], self.vocab)

        model = DummyModel(self.vocab)
        model.eval()
        loader = PyTorchDataLoader(dataset, batch_size=2)
        batch = next(iter(loader))
        model.forward(**batch)

    # This use case may seem a bit peculiar. It's intended for situations where
    # you have sparse inputs that are used as additional features for some
    # prediction, and they are sparse enough that they can be empty for some
    # cases. It would be silly to try to handle these as None in your model; it
    # makes a whole lot more sense to just have a minimally-sized tensor that
    # gets entirely masked and has no effect on the rest of the model.
    def test_batch_of_entirely_empty_lists_works(self):
        dataset = AllennlpDataset([self.empty_instance, self.empty_instance], self.vocab)

        model = DummyModel(self.vocab)
        model.eval()
        loader = PyTorchDataLoader(dataset, batch_size=2)
        batch = next(iter(loader))
        model.forward(**batch)

    def test_list_of_text_padding(self):
        from allennlp.data.token_indexers import PretrainedTransformerIndexer
        from allennlp.data.tokenizers import Token
        from allennlp.data.fields import (
            TextField,
            ListField,
        )
        from allennlp.data import Vocabulary

        word_indexer = {"tokens": PretrainedTransformerIndexer("albert-base-v2")}
        text_field = TextField(
            [
                Token(t, text_id=2, type_id=1)
                for t in ["▁allen", "n", "lp", "▁has", "▁no", "▁bugs", "."]
            ],
            word_indexer,
        )
        list_field = ListField([text_field])

        vocab = Vocabulary()
        list_field.index(vocab)

        padding_lengths = {
            "list_tokens___mask": 10,
            "list_tokens___token_ids": 10,
            "list_tokens___type_ids": 10,
            "num_fields": 2,
        }

        tensors = list_field.as_tensor(padding_lengths)["tokens"]
        assert tensors["mask"].size() == (2, 10)
        assert tensors["mask"][0, 0] == True  # noqa: E712
        assert tensors["mask"][0, 9] == False  # noqa: E712
        assert (tensors["mask"][1, :] == False).all()  # noqa: E712

        assert tensors["token_ids"].size() == (2, 10)
        assert tensors["token_ids"][0, 0] == 2
        assert tensors["token_ids"][0, 9] == 0
        assert (tensors["token_ids"][1, :] == 0).all()

        assert tensors["type_ids"].size() == (2, 10)
        assert tensors["type_ids"][0, 0] == 1
        assert tensors["type_ids"][0, 9] == 0
        assert (tensors["type_ids"][1, :] == 0).all()
