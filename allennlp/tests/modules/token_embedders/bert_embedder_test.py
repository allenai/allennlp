# pylint: disable=no-self-use,invalid-name
import torch

from pytorch_pretrained_bert.modeling import BertConfig, BertModel

from allennlp.common.testing import ModelTestCase
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField, ListField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers.wordpiece_indexer import PretrainedBertIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import BertBasicWordSplitter
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders.bert_token_embedder import BertEmbedder


class TestBertEmbedder(ModelTestCase):
    def setUp(self):
        super().setUp()

        vocab_path = self.FIXTURES_ROOT / 'bert' / 'vocab.txt'
        self.token_indexer = PretrainedBertIndexer(str(vocab_path))

        config_path = self.FIXTURES_ROOT / 'bert' / 'config.json'
        config = BertConfig(str(config_path))
        self.bert_model = BertModel(config)
        self.token_embedder = BertEmbedder(self.bert_model)

    def test_without_offsets(self):
        input_ids = torch.LongTensor([[3, 5, 9, 1, 2], [1, 5, 0, 0, 0]])
        result = self.token_embedder(input_ids)

        assert list(result.shape) == [2, 5, 12]

    def test_with_offsets(self):
        input_ids = torch.LongTensor([[3, 5, 9, 1, 2], [1, 5, 0, 0, 0]])
        offsets = torch.LongTensor([[0, 2, 4], [1, 0, 0]])

        result = self.token_embedder(input_ids, offsets=offsets)

        assert list(result.shape) == [2, 3, 12]

    def test_end_to_end(self):
        tokenizer = WordTokenizer(word_splitter=BertBasicWordSplitter())

        #            2   3    4   3     5     6   8      9    2   14   12
        sentence1 = "the quickest quick brown fox jumped over the lazy dog"
        tokens1 = tokenizer.tokenize(sentence1)

        #            2   3     5     6   8      9    2  15 10 11 14   1
        sentence2 = "the quick brown fox jumped over the laziest lazy elmo"
        tokens2 = tokenizer.tokenize(sentence2)

        vocab = Vocabulary()

        instance1 = Instance({"tokens": TextField(tokens1, {"bert": self.token_indexer})})
        instance2 = Instance({"tokens": TextField(tokens2, {"bert": self.token_indexer})})

        batch = Batch([instance1, instance2])
        batch.index_instances(vocab)

        padding_lengths = batch.get_padding_lengths()
        tensor_dict = batch.as_tensor_dict(padding_lengths)
        tokens = tensor_dict["tokens"]

        # 16 = [CLS], 17 = [SEP]
        assert tokens["bert"].tolist() == [[16, 2, 3, 4, 3, 5, 6, 8, 9, 2, 14, 12, 17, 0],
                                           [16, 2, 3, 5, 6, 8, 9, 2, 15, 10, 11, 14, 1, 17]]

        assert tokens["bert-offsets"].tolist() == [[1, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                                                   [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]]

        # No offsets, should get 14 vectors back ([CLS] + 12 token wordpieces + [SEP])
        bert_vectors = self.token_embedder(tokens["bert"])
        assert list(bert_vectors.shape) == [2, 14, 12]

        # Offsets, should get 10 vectors back.
        bert_vectors = self.token_embedder(tokens["bert"], offsets=tokens["bert-offsets"])
        assert list(bert_vectors.shape) == [2, 10, 12]

        # Now try top_layer_only = True
        tlo_embedder = BertEmbedder(self.bert_model, top_layer_only=True)
        bert_vectors = tlo_embedder(tokens["bert"])
        assert list(bert_vectors.shape) == [2, 14, 12]

        bert_vectors = tlo_embedder(tokens["bert"], offsets=tokens["bert-offsets"])
        assert list(bert_vectors.shape) == [2, 10, 12]

    def test_padding_for_equal_length_indices(self):
        tokenizer = WordTokenizer(word_splitter=BertBasicWordSplitter())

        #            2   3     5     6   8      9    2   14   12
        sentence = "the quick brown fox jumped over the lazy dog"
        tokens = tokenizer.tokenize(sentence)

        vocab = Vocabulary()

        instance = Instance({"tokens": TextField(tokens, {"bert": self.token_indexer})})

        batch = Batch([instance])
        batch.index_instances(vocab)

        padding_lengths = batch.get_padding_lengths()
        tensor_dict = batch.as_tensor_dict(padding_lengths)
        tokens = tensor_dict["tokens"]

        assert tokens["bert"].tolist() == [[16, 2, 3, 5, 6, 8, 9, 2, 14, 12, 17]]

        assert tokens["bert-offsets"].tolist() == [[1, 2, 3, 4, 5, 6, 7, 8, 9]]

    def test_squad_with_unwordpieceable_passage(self):
        # pylint: disable=line-too-long
        tokenizer = WordTokenizer()

        token_indexer = PretrainedBertIndexer("bert-base-uncased")

        passage1 = ("There were four major HDTV systems tested by SMPTE in the late 1970s, "
                    "and in 1979 an SMPTE study group released A Study of High Definition Television Systems:")
        question1 = "Who released A Study of High Definition Television Systems?"

        passage2 = ("Broca, being what today would be called a neurosurgeon, "
                    "had taken an interest in the pathology of speech. He wanted "
                    "to localize the difference between man and the other animals, "
                    "which appeared to reside in speech. He discovered the speech "
                    "center of the human brain, today called Broca's area after him. "
                    "His interest was mainly in Biological anthropology, but a German "
                    "philosopher specializing in psychology, Theodor Waitz, took up the "
                    "theme of general and social anthropology in his six-volume work, "
                    "entitled Die Anthropologie der Naturvölker, 1859–1864. The title was "
                    """soon translated as "The Anthropology of Primitive Peoples". """
                    "The last two volumes were published posthumously.")
        question2 = "What did Broca discover in the human brain?"

        from allennlp.data.dataset_readers.reading_comprehension.util import make_reading_comprehension_instance

        instance1 = make_reading_comprehension_instance(tokenizer.tokenize(question1),
                                                        tokenizer.tokenize(passage1),
                                                        {"bert": token_indexer},
                                                        passage1)

        instance2 = make_reading_comprehension_instance(tokenizer.tokenize(question2),
                                                        tokenizer.tokenize(passage2),
                                                        {"bert": token_indexer},
                                                        passage2)

        vocab = Vocabulary()

        batch = Batch([instance1, instance2])
        batch.index_instances(vocab)

        padding_lengths = batch.get_padding_lengths()
        tensor_dict = batch.as_tensor_dict(padding_lengths)
        qtokens = tensor_dict["question"]
        ptokens = tensor_dict["passage"]

        config = BertConfig(len(token_indexer.vocab))
        model = BertModel(config)
        embedder = BertEmbedder(model)

        _ = embedder(ptokens["bert"], offsets=ptokens["bert-offsets"])
        _ = embedder(qtokens["bert"], offsets=qtokens["bert-offsets"])

    def test_max_length(self):
        config = BertConfig(len(self.token_indexer.vocab))
        model = BertModel(config)
        embedder = BertEmbedder(model)

        tokenizer = WordTokenizer(word_splitter=BertBasicWordSplitter())
        sentence = "the " * 1000
        tokens = tokenizer.tokenize(sentence)

        vocab = Vocabulary()

        instance = Instance({"tokens": TextField(tokens, {"bert": self.token_indexer})})

        batch = Batch([instance])
        batch.index_instances(vocab)

        padding_lengths = batch.get_padding_lengths()
        tensor_dict = batch.as_tensor_dict(padding_lengths)
        tokens = tensor_dict["tokens"]
        embedder(tokens["bert"], tokens["bert-offsets"])

    def test_end_to_end_with_higher_order_inputs(self):
        tokenizer = WordTokenizer(word_splitter=BertBasicWordSplitter())

        #            2   3    4   3     5     6   8      9    2   14   12
        sentence1 = "the quickest quick brown fox jumped over the lazy dog"
        tokens1 = tokenizer.tokenize(sentence1)
        text_field1 = TextField(tokens1, {"bert": self.token_indexer})

        #            2   3     5     6   8      9    2  15 10 11 14   1
        sentence2 = "the quick brown fox jumped over the laziest lazy elmo"
        tokens2 = tokenizer.tokenize(sentence2)
        text_field2 = TextField(tokens2, {"bert": self.token_indexer})

        #            2   5    15 10 11 6
        sentence3 = "the brown laziest fox"
        tokens3 = tokenizer.tokenize(sentence3)
        text_field3 = TextField(tokens3, {"bert": self.token_indexer})

        vocab = Vocabulary()

        instance1 = Instance({"tokens": ListField([text_field1])})
        instance2 = Instance({"tokens": ListField([text_field2, text_field3])})

        batch = Batch([instance1, instance2])
        batch.index_instances(vocab)

        padding_lengths = batch.get_padding_lengths()
        tensor_dict = batch.as_tensor_dict(padding_lengths, verbose=True)
        tokens = tensor_dict["tokens"]

        # No offsets, should get 14 vectors back ([CLS] + 12 wordpieces + [SEP])
        bert_vectors = self.token_embedder(tokens["bert"])
        assert list(bert_vectors.shape) == [2, 2, 14, 12]

        # Offsets, should get 10 vectors back.
        bert_vectors = self.token_embedder(tokens["bert"], offsets=tokens["bert-offsets"])
        assert list(bert_vectors.shape) == [2, 2, 10, 12]

        # Now try top_layer_only = True
        tlo_embedder = BertEmbedder(self.bert_model, top_layer_only=True)
        bert_vectors = tlo_embedder(tokens["bert"])
        assert list(bert_vectors.shape) == [2, 2, 14, 12]

        bert_vectors = tlo_embedder(tokens["bert"], offsets=tokens["bert-offsets"])
        assert list(bert_vectors.shape) == [2, 2, 10, 12]

    def test_sliding_window(self):
        tokenizer = WordTokenizer(word_splitter=BertBasicWordSplitter())

        sentence = "the quickest quick brown fox jumped over the lazy dog"
        tokens = tokenizer.tokenize(sentence)

        vocab = Vocabulary()

        vocab_path = self.FIXTURES_ROOT / 'bert' / 'vocab.txt'
        token_indexer = PretrainedBertIndexer(str(vocab_path), truncate_long_sequences=False, max_pieces=8)

        config_path = self.FIXTURES_ROOT / 'bert' / 'config.json'
        config = BertConfig(str(config_path))
        bert_model = BertModel(config)
        token_embedder = BertEmbedder(bert_model, max_pieces=8)

        instance = Instance({"tokens": TextField(tokens, {"bert": token_indexer})})

        batch = Batch([instance])
        batch.index_instances(vocab)

        padding_lengths = batch.get_padding_lengths()
        tensor_dict = batch.as_tensor_dict(padding_lengths)
        tokens = tensor_dict["tokens"]

        # 16 = [CLS], 17 = [SEP]
        # 1 full window + 1 half window with start/end tokens
        assert tokens["bert"].tolist() == [[16, 2, 3, 4, 3, 5, 6, 17,
                                            16, 3, 5, 6, 8, 9, 2, 17,
                                            16, 8, 9, 2, 14, 12, 17]]
        assert tokens["bert-offsets"].tolist() == [[1, 3, 4, 5, 6, 7, 8, 9, 10, 11]]

        bert_vectors = token_embedder(tokens["bert"])
        assert list(bert_vectors.shape) == [1, 13, 12]

        # Testing without token_type_ids
        bert_vectors = token_embedder(tokens["bert"],
                                      offsets=tokens["bert-offsets"])
        assert list(bert_vectors.shape) == [1, 10, 12]

        # Testing with token_type_ids
        bert_vectors = token_embedder(tokens["bert"],
                                      offsets=tokens["bert-offsets"],
                                      token_type_ids=tokens["bert-type-ids"])
        assert list(bert_vectors.shape) == [1, 10, 12]

    def test_sliding_window_with_batch(self):
        tokenizer = WordTokenizer(word_splitter=BertBasicWordSplitter())

        sentence = "the quickest quick brown fox jumped over the lazy dog"
        tokens = tokenizer.tokenize(sentence)

        vocab = Vocabulary()

        vocab_path = self.FIXTURES_ROOT / 'bert' / 'vocab.txt'
        token_indexer = PretrainedBertIndexer(str(vocab_path), truncate_long_sequences=False, max_pieces=8)

        config_path = self.FIXTURES_ROOT / 'bert' / 'config.json'
        config = BertConfig(str(config_path))
        bert_model = BertModel(config)
        token_embedder = BertEmbedder(bert_model, max_pieces=8)

        instance = Instance({"tokens": TextField(tokens, {"bert": token_indexer})})
        instance2 = Instance({"tokens": TextField(tokens + tokens + tokens, {"bert": token_indexer})})

        batch = Batch([instance, instance2])
        batch.index_instances(vocab)

        padding_lengths = batch.get_padding_lengths()
        tensor_dict = batch.as_tensor_dict(padding_lengths)
        tokens = tensor_dict["tokens"]

        # Testing without token_type_ids
        bert_vectors = token_embedder(tokens["bert"],
                                      offsets=tokens["bert-offsets"])
        assert bert_vectors is not None

        # Testing with token_type_ids
        bert_vectors = token_embedder(tokens["bert"],
                                      offsets=tokens["bert-offsets"],
                                      token_type_ids=tokens["bert-type-ids"])
        assert bert_vectors is not None
