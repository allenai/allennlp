import codecs
import copy
import gzip
import pickle
import shutil
import zipfile
from copy import deepcopy

import pytest

from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Instance, Token
from allennlp.data.batch import Batch
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.tokenizers import CharacterTokenizer
from allennlp.data.vocabulary import (
    _NamespaceDependentDefaultDict,
    _read_pretrained_tokens,
    DEFAULT_OOV_TOKEN,
    Vocabulary,
)
from allennlp.modules.token_embedders.embedding import format_embeddings_file_uri


class TestVocabulary(AllenNlpTestCase):
    def setUp(self):
        token_indexer = SingleIdTokenIndexer("tokens")
        text_field = TextField(
            [Token(t) for t in ["a", "a", "a", "a", "b", "b", "c", "c", "c"]],
            {"tokens": token_indexer},
        )
        self.instance = Instance({"text": text_field})
        self.dataset = Batch([self.instance])
        super().setUp()

    def test_pickling(self):
        vocab = Vocabulary.from_instances(self.dataset)

        pickled = pickle.dumps(vocab)
        unpickled = pickle.loads(pickled)

        assert dict(unpickled._index_to_token) == dict(vocab._index_to_token)
        assert dict(unpickled._token_to_index) == dict(vocab._token_to_index)
        assert unpickled._non_padded_namespaces == vocab._non_padded_namespaces
        assert unpickled._oov_token == vocab._oov_token
        assert unpickled._padding_token == vocab._padding_token
        assert unpickled._retained_counter == vocab._retained_counter

    def test_from_dataset_respects_max_vocab_size_single_int(self):
        max_vocab_size = 1
        vocab = Vocabulary.from_instances(self.dataset, max_vocab_size=max_vocab_size)
        words = vocab.get_index_to_token_vocabulary().values()
        # Additional 2 tokens are '@@PADDING@@' and '@@UNKNOWN@@' by default
        assert len(words) == max_vocab_size + 2

        vocab = Vocabulary.from_instances(self.dataset, min_count=None)
        words = vocab.get_index_to_token_vocabulary().values()
        assert len(words) == 5

    def test_from_dataset_respects_min_count(self):
        vocab = Vocabulary.from_instances(self.dataset, min_count={"tokens": 4})
        words = vocab.get_index_to_token_vocabulary().values()
        assert "a" in words
        assert "b" not in words
        assert "c" not in words

        vocab = Vocabulary.from_instances(self.dataset, min_count=None)
        words = vocab.get_index_to_token_vocabulary().values()
        assert "a" in words
        assert "b" in words
        assert "c" in words

    def test_from_dataset_respects_exclusive_embedding_file(self):
        embeddings_filename = str(self.TEST_DIR / "embeddings.gz")
        with gzip.open(embeddings_filename, "wb") as embeddings_file:
            embeddings_file.write("a 1.0 2.3 -1.0\n".encode("utf-8"))
            embeddings_file.write("b 0.1 0.4 -4.0\n".encode("utf-8"))

        vocab = Vocabulary.from_instances(
            self.dataset,
            min_count={"tokens": 4},
            pretrained_files={"tokens": embeddings_filename},
            only_include_pretrained_words=True,
        )
        words = vocab.get_index_to_token_vocabulary().values()
        assert "a" in words
        assert "b" not in words
        assert "c" not in words

        vocab = Vocabulary.from_instances(
            self.dataset,
            pretrained_files={"tokens": embeddings_filename},
            only_include_pretrained_words=True,
        )
        words = vocab.get_index_to_token_vocabulary().values()
        assert "a" in words
        assert "b" in words
        assert "c" not in words

    def test_from_dataset_respects_inclusive_embedding_file(self):
        embeddings_filename = str(self.TEST_DIR / "embeddings.gz")
        with gzip.open(embeddings_filename, "wb") as embeddings_file:
            embeddings_file.write("a 1.0 2.3 -1.0\n".encode("utf-8"))
            embeddings_file.write("b 0.1 0.4 -4.0\n".encode("utf-8"))

        vocab = Vocabulary.from_instances(
            self.dataset,
            min_count={"tokens": 4},
            pretrained_files={"tokens": embeddings_filename},
            only_include_pretrained_words=False,
        )
        words = vocab.get_index_to_token_vocabulary().values()
        assert "a" in words
        assert "b" in words
        assert "c" not in words

        vocab = Vocabulary.from_instances(
            self.dataset,
            pretrained_files={"tokens": embeddings_filename},
            only_include_pretrained_words=False,
        )
        words = vocab.get_index_to_token_vocabulary().values()
        assert "a" in words
        assert "b" in words
        assert "c" in words

    def test_add_word_to_index_gives_consistent_results(self):
        vocab = Vocabulary()
        initial_vocab_size = vocab.get_vocab_size()
        word_index = vocab.add_token_to_namespace("word")
        assert "word" in vocab.get_index_to_token_vocabulary().values()
        assert vocab.get_token_index("word") == word_index
        assert vocab.get_token_from_index(word_index) == "word"
        assert vocab.get_vocab_size() == initial_vocab_size + 1

        # Now add it again, and make sure nothing changes.
        vocab.add_token_to_namespace("word")
        assert "word" in vocab.get_index_to_token_vocabulary().values()
        assert vocab.get_token_index("word") == word_index
        assert vocab.get_token_from_index(word_index) == "word"
        assert vocab.get_vocab_size() == initial_vocab_size + 1

    def test_namespaces(self):
        vocab = Vocabulary()
        initial_vocab_size = vocab.get_vocab_size()
        word_index = vocab.add_token_to_namespace("word", namespace="1")
        assert "word" in vocab.get_index_to_token_vocabulary(namespace="1").values()
        assert vocab.get_token_index("word", namespace="1") == word_index
        assert vocab.get_token_from_index(word_index, namespace="1") == "word"
        assert vocab.get_vocab_size(namespace="1") == initial_vocab_size + 1

        # Now add it again, in a different namespace and a different word, and make sure it's like
        # new.
        word2_index = vocab.add_token_to_namespace("word2", namespace="2")
        word_index = vocab.add_token_to_namespace("word", namespace="2")
        assert "word" in vocab.get_index_to_token_vocabulary(namespace="2").values()
        assert "word2" in vocab.get_index_to_token_vocabulary(namespace="2").values()
        assert vocab.get_token_index("word", namespace="2") == word_index
        assert vocab.get_token_index("word2", namespace="2") == word2_index
        assert vocab.get_token_from_index(word_index, namespace="2") == "word"
        assert vocab.get_token_from_index(word2_index, namespace="2") == "word2"
        assert vocab.get_vocab_size(namespace="2") == initial_vocab_size + 2

    def test_namespace_dependent_default_dict(self):
        default_dict = _NamespaceDependentDefaultDict(["bar", "*baz"], lambda: 7, lambda: 3)
        # 'foo' is not a padded namespace
        assert default_dict["foo"] == 7
        # "baz" is a direct match with a padded namespace
        assert default_dict["baz"] == 3
        # the following match the wildcard "*baz"
        assert default_dict["bar"] == 3
        assert default_dict["foobaz"] == 3

    def test_unknown_token(self):

        # We're putting this behavior in a test so that the behavior is documented.  There is
        # solver code that depends in a small way on how we treat the unknown token, so any
        # breaking change to this behavior should break a test, so you know you've done something
        # that needs more consideration.
        vocab = Vocabulary()
        oov_token = vocab._oov_token
        oov_index = vocab.get_token_index(oov_token)
        assert oov_index == 1
        assert vocab.get_token_index("unseen word") == oov_index

    def test_set_from_file_reads_padded_files(self):

        vocab_filename = self.TEST_DIR / "vocab_file"
        with codecs.open(vocab_filename, "w", "utf-8") as vocab_file:
            vocab_file.write("<S>\n")
            vocab_file.write("</S>\n")
            vocab_file.write("<UNK>\n")
            vocab_file.write("a\n")
            vocab_file.write("tricky\x0bchar\n")
            vocab_file.write("word\n")
            vocab_file.write("another\n")

        vocab = Vocabulary()
        vocab.set_from_file(vocab_filename, is_padded=True, oov_token="<UNK>")

        assert vocab._oov_token == DEFAULT_OOV_TOKEN
        assert vocab.get_token_index("random string") == 3
        assert vocab.get_token_index("<S>") == 1
        assert vocab.get_token_index("</S>") == 2
        assert vocab.get_token_index(DEFAULT_OOV_TOKEN) == 3
        assert vocab.get_token_index("a") == 4
        assert vocab.get_token_index("tricky\x0bchar") == 5
        assert vocab.get_token_index("word") == 6
        assert vocab.get_token_index("another") == 7
        assert vocab.get_token_from_index(0) == vocab._padding_token
        assert vocab.get_token_from_index(1) == "<S>"
        assert vocab.get_token_from_index(2) == "</S>"
        assert vocab.get_token_from_index(3) == DEFAULT_OOV_TOKEN
        assert vocab.get_token_from_index(4) == "a"
        assert vocab.get_token_from_index(5) == "tricky\x0bchar"
        assert vocab.get_token_from_index(6) == "word"
        assert vocab.get_token_from_index(7) == "another"

    def test_set_from_file_reads_non_padded_files(self):

        vocab_filename = self.TEST_DIR / "vocab_file"
        with codecs.open(vocab_filename, "w", "utf-8") as vocab_file:
            vocab_file.write("B-PERS\n")
            vocab_file.write("I-PERS\n")
            vocab_file.write("O\n")
            vocab_file.write("B-ORG\n")
            vocab_file.write("I-ORG\n")

        vocab = Vocabulary()
        vocab.set_from_file(vocab_filename, is_padded=False, namespace="tags")
        assert vocab.get_token_index("B-PERS", namespace="tags") == 0
        assert vocab.get_token_index("I-PERS", namespace="tags") == 1
        assert vocab.get_token_index("O", namespace="tags") == 2
        assert vocab.get_token_index("B-ORG", namespace="tags") == 3
        assert vocab.get_token_index("I-ORG", namespace="tags") == 4
        assert vocab.get_token_from_index(0, namespace="tags") == "B-PERS"
        assert vocab.get_token_from_index(1, namespace="tags") == "I-PERS"
        assert vocab.get_token_from_index(2, namespace="tags") == "O"
        assert vocab.get_token_from_index(3, namespace="tags") == "B-ORG"
        assert vocab.get_token_from_index(4, namespace="tags") == "I-ORG"

    def test_saving_and_loading(self):

        vocab_dir = self.TEST_DIR / "vocab_save"

        vocab = Vocabulary(non_padded_namespaces=["a", "c"])
        vocab.add_tokens_to_namespace(
            ["a0", "a1", "a2"], namespace="a"
        )  # non-padded, should start at 0
        vocab.add_tokens_to_namespace(["b2", "b3"], namespace="b")  # padded, should start at 2

        vocab.save_to_files(vocab_dir)
        vocab2 = Vocabulary.from_files(vocab_dir)

        assert vocab2._non_padded_namespaces == {"a", "c"}

        # Check namespace a.
        assert vocab2.get_vocab_size(namespace="a") == 3
        assert vocab2.get_token_from_index(0, namespace="a") == "a0"
        assert vocab2.get_token_from_index(1, namespace="a") == "a1"
        assert vocab2.get_token_from_index(2, namespace="a") == "a2"
        assert vocab2.get_token_index("a0", namespace="a") == 0
        assert vocab2.get_token_index("a1", namespace="a") == 1
        assert vocab2.get_token_index("a2", namespace="a") == 2

        # Check namespace b.
        assert vocab2.get_vocab_size(namespace="b") == 4  # (unk + padding + two tokens)
        assert vocab2.get_token_from_index(0, namespace="b") == vocab._padding_token
        assert vocab2.get_token_from_index(1, namespace="b") == vocab._oov_token
        assert vocab2.get_token_from_index(2, namespace="b") == "b2"
        assert vocab2.get_token_from_index(3, namespace="b") == "b3"
        assert vocab2.get_token_index(vocab._padding_token, namespace="b") == 0
        assert vocab2.get_token_index(vocab._oov_token, namespace="b") == 1
        assert vocab2.get_token_index("b2", namespace="b") == 2
        assert vocab2.get_token_index("b3", namespace="b") == 3

        # Check the dictionaries containing the reverse mapping are identical.
        assert vocab.get_index_to_token_vocabulary("a") == vocab2.get_index_to_token_vocabulary("a")
        assert vocab.get_index_to_token_vocabulary("b") == vocab2.get_index_to_token_vocabulary("b")

    def test_saving_and_loading_works_with_byte_encoding(self):
        # We're going to set a vocabulary from a TextField using byte encoding, index it, save the
        # vocab, load the vocab, then index the text field again, and make sure we get the same
        # result.
        tokenizer = CharacterTokenizer(byte_encoding="utf-8")
        token_indexer = TokenCharactersIndexer(character_tokenizer=tokenizer, min_padding_length=2)
        tokens = [Token(t) for t in ["Øyvind", "für", "汉字"]]
        text_field = TextField(tokens, {"characters": token_indexer})
        dataset = Batch([Instance({"sentence": text_field})])
        vocab = Vocabulary.from_instances(dataset)
        text_field.index(vocab)
        indexed_tokens = deepcopy(text_field._indexed_tokens)

        vocab_dir = self.TEST_DIR / "vocab_save"
        vocab.save_to_files(vocab_dir)
        vocab2 = Vocabulary.from_files(vocab_dir)
        text_field2 = TextField(tokens, {"characters": token_indexer})
        text_field2.index(vocab2)
        indexed_tokens2 = deepcopy(text_field2._indexed_tokens)
        assert indexed_tokens == indexed_tokens2

    def test_from_params(self):
        # Save a vocab to check we can load it from_params.
        vocab_dir = self.TEST_DIR / "vocab_save"
        vocab = Vocabulary(non_padded_namespaces=["a", "c"])
        vocab.add_tokens_to_namespace(
            ["a0", "a1", "a2"], namespace="a"
        )  # non-padded, should start at 0
        vocab.add_tokens_to_namespace(["b2", "b3"], namespace="b")  # padded, should start at 2
        vocab.save_to_files(vocab_dir)

        params = Params({"type": "from_files", "directory": vocab_dir})
        vocab2 = Vocabulary.from_params(params)
        assert vocab.get_index_to_token_vocabulary("a") == vocab2.get_index_to_token_vocabulary("a")
        assert vocab.get_index_to_token_vocabulary("b") == vocab2.get_index_to_token_vocabulary("b")

        # Test case where we build a vocab from a dataset.
        vocab2 = Vocabulary.from_params(Params({}), instances=self.dataset)
        assert vocab2.get_index_to_token_vocabulary("tokens") == {
            0: "@@PADDING@@",
            1: "@@UNKNOWN@@",
            2: "a",
            3: "c",
            4: "b",
        }
        # Test from_params raises when we have neither a dataset and a vocab_directory.
        with pytest.raises(ConfigurationError):
            _ = Vocabulary.from_params(Params({}))

        # Test from_params raises when there are any other dict keys
        # present apart from 'directory' and we aren't calling from_dataset.
        with pytest.raises(ConfigurationError):
            _ = Vocabulary.from_params(
                Params({"type": "from_files", "directory": vocab_dir, "min_count": {"tokens": 2}})
            )

    def test_from_params_adds_tokens_to_vocab(self):
        vocab = Vocabulary.from_params(
            Params({"tokens_to_add": {"tokens": ["q", "x", "z"]}}), instances=self.dataset
        )
        assert vocab.get_index_to_token_vocabulary("tokens") == {
            0: "@@PADDING@@",
            1: "@@UNKNOWN@@",
            2: "a",
            3: "c",
            4: "b",
            5: "q",
            6: "x",
            7: "z",
        }

    def test_valid_vocab_extension(self):
        vocab_dir = self.TEST_DIR / "vocab_save"
        # Test: padded/non-padded common namespaces are extending appropriately
        non_padded_namespaces_list = [[], ["tokens"]]
        for non_padded_namespaces in non_padded_namespaces_list:
            original_vocab = Vocabulary(non_padded_namespaces=non_padded_namespaces)
            original_vocab.add_tokens_to_namespace(["d", "a", "b"], namespace="tokens")
            text_field = TextField(
                [Token(t) for t in ["a", "d", "c", "e"]], {"tokens": SingleIdTokenIndexer("tokens")}
            )
            vocab_dir = self.TEST_DIR / "vocab_save"
            shutil.rmtree(vocab_dir, ignore_errors=True)
            original_vocab.save_to_files(vocab_dir)
            instances = Batch([Instance({"text": text_field})])
            params = Params(
                {
                    "type": "extend",
                    "directory": vocab_dir,
                    "non_padded_namespaces": non_padded_namespaces,
                }
            )
            extended_vocab = Vocabulary.from_params(params, instances=instances)

            extra_count = 2 if extended_vocab.is_padded("tokens") else 0
            assert extended_vocab.get_token_index("d", "tokens") == 0 + extra_count
            assert extended_vocab.get_token_index("a", "tokens") == 1 + extra_count
            assert extended_vocab.get_token_index("b", "tokens") == 2 + extra_count

            assert extended_vocab.get_token_index("c", "tokens")  # should be present
            assert extended_vocab.get_token_index("e", "tokens")  # should be present

            assert extended_vocab.get_vocab_size("tokens") == 5 + extra_count

        # Test: padded/non-padded non-common namespaces are extending appropriately
        non_padded_namespaces_list = [[], ["tokens1"], ["tokens1", "tokens2"]]
        for non_padded_namespaces in non_padded_namespaces_list:
            original_vocab = Vocabulary(non_padded_namespaces=non_padded_namespaces)
            original_vocab.add_token_to_namespace("a", namespace="tokens1")  # index2
            text_field = TextField(
                [Token(t) for t in ["b"]], {"tokens2": SingleIdTokenIndexer("tokens2")}
            )
            instances = Batch([Instance({"text": text_field})])
            vocab_dir = self.TEST_DIR / "vocab_save"
            shutil.rmtree(vocab_dir, ignore_errors=True)
            original_vocab.save_to_files(vocab_dir)

            params = Params(
                {
                    "type": "extend",
                    "directory": vocab_dir,
                    "non_padded_namespaces": non_padded_namespaces,
                }
            )
            extended_vocab = Vocabulary.from_params(params, instances=instances)

            # Should have two namespaces
            assert len(extended_vocab._token_to_index) == 2

            extra_count = 2 if extended_vocab.is_padded("tokens1") else 0
            assert extended_vocab.get_vocab_size("tokens1") == 1 + extra_count

            extra_count = 2 if extended_vocab.is_padded("tokens2") else 0
            assert extended_vocab.get_vocab_size("tokens2") == 1 + extra_count

    def test_invalid_vocab_extension(self):
        vocab_dir = self.TEST_DIR / "vocab_save"
        original_vocab = Vocabulary(non_padded_namespaces=["tokens1"])
        original_vocab.add_tokens_to_namespace(["a", "b"], namespace="tokens1")
        original_vocab.add_token_to_namespace("p", namespace="tokens2")
        original_vocab.save_to_files(vocab_dir)
        text_field1 = TextField(
            [Token(t) for t in ["a", "c"]], {"tokens1": SingleIdTokenIndexer("tokens1")}
        )
        text_field2 = TextField(
            [Token(t) for t in ["p", "q", "r"]], {"tokens2": SingleIdTokenIndexer("tokens2")}
        )
        instances = Batch([Instance({"text1": text_field1, "text2": text_field2})])

        # Following 2 should give error: tokens1 is non-padded in original_vocab but not in instances
        params = Params(
            {
                "type": "extend",
                "directory": vocab_dir,
                "non_padded_namespaces": [],
                "tokens_to_add": {"tokens1": ["a"], "tokens2": ["p"]},
            }
        )
        with pytest.raises(ConfigurationError):
            _ = Vocabulary.from_params(params, instances=instances)

        # Following 2 should not give error: overlapping namespaces have same padding setting
        params = Params(
            {
                "type": "extend",
                "directory": vocab_dir,
                "non_padded_namespaces": ["tokens1"],
                "tokens_to_add": {"tokens1": ["a"], "tokens2": ["p"]},
            }
        )
        Vocabulary.from_params(params, instances=instances)

        # Following 2 should give error: tokens2 is padded in instances but not in original_vocab
        params = Params(
            {
                "type": "extend",
                "directory": vocab_dir,
                "non_padded_namespaces": ["tokens1", "tokens2"],
                "tokens_to_add": {"tokens1": ["a"], "tokens2": ["p"]},
            }
        )
        with pytest.raises(ConfigurationError):
            _ = Vocabulary.from_params(params, instances=instances)

    def test_from_params_extend_config(self):

        vocab_dir = self.TEST_DIR / "vocab_save"
        original_vocab = Vocabulary(non_padded_namespaces=["tokens"])
        original_vocab.add_token_to_namespace("a", namespace="tokens")
        original_vocab.save_to_files(vocab_dir)

        text_field = TextField(
            [Token(t) for t in ["a", "b"]], {"tokens": SingleIdTokenIndexer("tokens")}
        )
        instances = Batch([Instance({"text": text_field})])

        # If you ask to extend vocab from `directory`, instances must be passed
        # in Vocabulary constructor, or else there is nothing to extend to.
        params = Params({"type": "extend", "directory": vocab_dir})
        with pytest.raises(ConfigurationError):
            _ = Vocabulary.from_params(params)

        # If you ask to extend vocab, `directory` key must be present in params,
        # or else there is nothing to extend from.
        params = Params({"type": "extend"})
        with pytest.raises(ConfigurationError):
            _ = Vocabulary.from_params(params, instances=instances)

    def test_from_params_valid_vocab_extension_thoroughly(self):
        """
        Tests for Valid Vocab Extension thoroughly: Vocab extension is valid
        when overlapping namespaces have same padding behaviour (padded/non-padded)
        Summary of namespace paddings in this test:
        original_vocab namespaces
            tokens0     padded
            tokens1     non-padded
            tokens2     padded
            tokens3     non-padded
        instances namespaces
            tokens0     padded
            tokens1     non-padded
            tokens4     padded
            tokens5     non-padded
        TypicalExtention example: (of tokens1 namespace)
        -> original_vocab index2token
           apple          #0->apple
           bat            #1->bat
           cat            #2->cat
        -> Token to be extended with: cat, an, apple, banana, atom, bat
        -> extended_vocab: index2token
           apple           #0->apple
           bat             #1->bat
           cat             #2->cat
           an              #3->an
           atom            #4->atom
           banana          #5->banana
        """

        vocab_dir = self.TEST_DIR / "vocab_save"
        original_vocab = Vocabulary(non_padded_namespaces=["tokens1", "tokens3"])
        original_vocab.add_token_to_namespace("apple", namespace="tokens0")  # index:2
        original_vocab.add_token_to_namespace("bat", namespace="tokens0")  # index:3
        original_vocab.add_token_to_namespace("cat", namespace="tokens0")  # index:4

        original_vocab.add_token_to_namespace("apple", namespace="tokens1")  # index:0
        original_vocab.add_token_to_namespace("bat", namespace="tokens1")  # index:1
        original_vocab.add_token_to_namespace("cat", namespace="tokens1")  # index:2

        original_vocab.add_token_to_namespace("a", namespace="tokens2")  # index:0
        original_vocab.add_token_to_namespace("b", namespace="tokens2")  # index:1
        original_vocab.add_token_to_namespace("c", namespace="tokens2")  # index:2

        original_vocab.add_token_to_namespace("p", namespace="tokens3")  # index:0
        original_vocab.add_token_to_namespace("q", namespace="tokens3")  # index:1

        original_vocab.save_to_files(vocab_dir)

        text_field0 = TextField(
            [Token(t) for t in ["cat", "an", "apple", "banana", "atom", "bat"]],
            {"tokens0": SingleIdTokenIndexer("tokens0")},
        )
        text_field1 = TextField(
            [Token(t) for t in ["cat", "an", "apple", "banana", "atom", "bat"]],
            {"tokens1": SingleIdTokenIndexer("tokens1")},
        )
        text_field4 = TextField(
            [Token(t) for t in ["l", "m", "n", "o"]], {"tokens4": SingleIdTokenIndexer("tokens4")}
        )
        text_field5 = TextField(
            [Token(t) for t in ["x", "y", "z"]], {"tokens5": SingleIdTokenIndexer("tokens5")}
        )
        instances = Batch(
            [
                Instance(
                    {
                        "text0": text_field0,
                        "text1": text_field1,
                        "text4": text_field4,
                        "text5": text_field5,
                    }
                )
            ]
        )

        params = Params(
            {
                "type": "extend",
                "directory": vocab_dir,
                "non_padded_namespaces": ["tokens1", "tokens5"],
            }
        )
        extended_vocab = Vocabulary.from_params(params, instances=instances)

        # namespaces: tokens0, tokens1 is common.
        # tokens2, tokens3 only vocab has. tokens4, tokens5 only instances
        extended_namespaces = {*extended_vocab._token_to_index}
        assert extended_namespaces == {"tokens{}".format(i) for i in range(6)}

        # # Check that _non_padded_namespaces list is consistent after extension
        assert extended_vocab._non_padded_namespaces == {"tokens1", "tokens3", "tokens5"}

        # # original_vocab["tokens1"] has 3 tokens, instances of "tokens1" ns has 5 tokens. 2 overlapping
        assert extended_vocab.get_vocab_size("tokens1") == 6
        assert extended_vocab.get_vocab_size("tokens0") == 8  # 2 extra overlapping because padded

        # namespace tokens3, tokens4 was only in original_vocab,
        # and its token count should be same in extended_vocab
        assert extended_vocab.get_vocab_size("tokens2") == original_vocab.get_vocab_size("tokens2")
        assert extended_vocab.get_vocab_size("tokens3") == original_vocab.get_vocab_size("tokens3")

        # namespace tokens2 was only in instances,
        # and its token count should be same in extended_vocab
        assert extended_vocab.get_vocab_size("tokens4") == 6  # l,m,n,o + oov + padding
        assert extended_vocab.get_vocab_size("tokens5") == 3  # x,y,z

        # Word2index mapping of all words in all namespaces of original_vocab
        # should be maintained in extended_vocab
        for namespace, token2index in original_vocab._token_to_index.items():
            for token, _ in token2index.items():
                vocab_index = original_vocab.get_token_index(token, namespace)
                extended_vocab_index = extended_vocab.get_token_index(token, namespace)
                assert vocab_index == extended_vocab_index
        # And same for Index2Word mapping
        for namespace, index2token in original_vocab._index_to_token.items():
            for index, _ in index2token.items():
                vocab_token = original_vocab.get_token_from_index(index, namespace)
                extended_vocab_token = extended_vocab.get_token_from_index(index, namespace)
                assert vocab_token == extended_vocab_token

        # Manual Print Check
        # original_vocab._token_to_index :>
        # {
        #   "tokens0": {"@@PADDING@@":0,"@@UNKNOWN@@":1,"apple":2,"bat":3,"cat":4},
        #   "tokens1": {"apple": 0,"bat":1,"cat":2},
        #   "tokens2": {"@@PADDING@@":0,"@@UNKNOWN@@":1,"a":2,"b":3,"c": 4},
        #   "tokens3": {"p":0,"q":1}
        # }
        # extended_vocab._token_to_index :>
        # {
        #   "tokens0": {"@@PADDING@@": 0,"@@UNKNOWN@@": 1,
        #               "apple": 2,"bat": 3,"cat": 4,"an": 5,"banana": 6,"atom": 7},
        #   "tokens1": {"apple": 0,"bat": 1,"cat": 2,"an": 3,"banana": 4,"atom": 5},
        #   "tokens2": {"@@PADDING@@": 0,"@@UNKNOWN@@": 1,"a": 2,"b": 3,"c": 4},
        #   "tokens3": {"p": 0,"q": 1},
        #   "tokens4": {"@@PADDING@@": 0,"@@UNKNOWN@@": 1,"l": 2,"m": 3,"n": 4,"o": 5},
        #   "tokens5": {"x": 0,"y": 1,"z": 2}
        # }

    def test_vocab_can_print(self):
        vocab = Vocabulary(non_padded_namespaces=["a", "c"])
        vocab.add_tokens_to_namespace(["a0", "a1", "a2"], namespace="a")
        vocab.add_tokens_to_namespace(["b2", "b3"], namespace="b")
        print(vocab)

    def test_read_pretrained_words(self):
        # The fixture "fake_embeddings.5d.txt" was generated using the words in this random quote
        words = set(
            "If you think you are too small to make a difference "
            "try to sleeping with a mosquito àèìòù".split(" ")
        )

        # Reading from a single (compressed) file or a single-file archive
        base_path = str(self.FIXTURES_ROOT / "embeddings/fake_embeddings.5d.txt")
        for ext in ["", ".gz", ".lzma", ".bz2", ".zip", ".tar.gz"]:
            file_path = base_path + ext
            words_read = set(_read_pretrained_tokens(file_path))
            assert words_read == words, (
                f"Wrong words for file {file_path}\n"
                f"   Read: {sorted(words_read)}\n"
                f"Correct: {sorted(words)}"
            )

        # Reading from a multi-file archive
        base_path = str(self.FIXTURES_ROOT / "embeddings/multi-file-archive")
        file_path = "folder/fake_embeddings.5d.txt"
        for ext in [".zip", ".tar.gz"]:
            archive_path = base_path + ext
            embeddings_file_uri = format_embeddings_file_uri(archive_path, file_path)
            words_read = set(_read_pretrained_tokens(embeddings_file_uri))
            assert words_read == words, (
                f"Wrong words for file {archive_path}\n"
                f"   Read: {sorted(words_read)}\n"
                f"Correct: {sorted(words)}"
            )

    def test_from_instances_exclusive_embeddings_file_inside_archive(self):
        """ Just for ensuring there are no problems when reading pretrained tokens from an archive """
        # Read embeddings file from archive
        archive_path = str(self.TEST_DIR / "embeddings-archive.zip")

        with zipfile.ZipFile(archive_path, "w") as archive:
            file_path = "embedding.3d.vec"
            with archive.open(file_path, "w") as embeddings_file:
                embeddings_file.write("a 1.0 2.3 -1.0\n".encode("utf-8"))
                embeddings_file.write("b 0.1 0.4 -4.0\n".encode("utf-8"))

            with archive.open("dummy.vec", "w") as dummy_file:
                dummy_file.write("c 1.0 2.3 -1.0 3.0\n".encode("utf-8"))

        embeddings_file_uri = format_embeddings_file_uri(archive_path, file_path)
        vocab = Vocabulary.from_instances(
            self.dataset,
            min_count={"tokens": 4},
            pretrained_files={"tokens": embeddings_file_uri},
            only_include_pretrained_words=True,
        )

        words = set(vocab.get_index_to_token_vocabulary().values())
        assert "a" in words
        assert "b" not in words
        assert "c" not in words

        vocab = Vocabulary.from_instances(
            self.dataset,
            pretrained_files={"tokens": embeddings_file_uri},
            only_include_pretrained_words=True,
        )
        words = set(vocab.get_index_to_token_vocabulary().values())
        assert "a" in words
        assert "b" in words
        assert "c" not in words

    def test_registrability(self):
        @Vocabulary.register("my-vocabulary", constructor="constructor")
        class MyVocabulary(Vocabulary):
            @classmethod
            def constructor(cls):
                return MyVocabulary()

        params = Params({"type": "my-vocabulary"})

        instance = Instance(fields={})

        vocab = Vocabulary.from_params(params=params, instances=[instance])

        assert isinstance(vocab, MyVocabulary)

    def test_max_vocab_size_dict(self):
        params = Params({"max_vocab_size": {"tokens": 1, "characters": 20}})

        vocab = Vocabulary.from_params(params=params, instances=self.dataset)
        words = vocab.get_index_to_token_vocabulary().values()
        # Additional 2 tokens are '@@PADDING@@' and '@@UNKNOWN@@' by default
        assert len(words) == 3

    def test_max_vocab_size_partial_dict(self):
        indexers = {
            "tokens": SingleIdTokenIndexer(),
            "token_characters": TokenCharactersIndexer(min_padding_length=3),
        }
        instance = Instance(
            {
                "text": TextField(
                    [Token(w) for w in "Abc def ghi jkl mno pqr stu vwx yz".split(" ")], indexers
                )
            }
        )
        dataset = Batch([instance])
        params = Params({"max_vocab_size": {"tokens": 1}})

        vocab = Vocabulary.from_params(params=params, instances=dataset)
        assert len(vocab.get_index_to_token_vocabulary("tokens").values()) == 3  # 1 + 2
        assert len(vocab.get_index_to_token_vocabulary("token_characters").values()) == 28  # 26 + 2

    def test_min_pretrained_embeddings(self):
        params = Params(
            {
                "pretrained_files": {
                    "tokens": str(self.FIXTURES_ROOT / "embeddings/glove.6B.100d.sample.txt.gz")
                },
                "min_pretrained_embeddings": {"tokens": 50},
            }
        )

        vocab = Vocabulary.from_params(params=params, instances=self.dataset)
        assert vocab.get_vocab_size() >= 50
        assert vocab.get_token_index("his") > 1  # not @@UNKNOWN@@

    def test_custom_padding_oov_tokens(self):
        vocab = Vocabulary(oov_token="[UNK]")
        assert vocab._oov_token == "[UNK]"
        assert vocab._padding_token == "@@PADDING@@"

        vocab = Vocabulary(padding_token="[PAD]")
        assert vocab._oov_token == "@@UNKNOWN@@"
        assert vocab._padding_token == "[PAD]"

        vocab_dir = self.TEST_DIR / "vocab_save"
        vocab = Vocabulary(oov_token="<UNK>")
        vocab.add_tokens_to_namespace(["a0", "a1", "a2"], namespace="a")
        vocab.save_to_files(vocab_dir)

        params = Params({"type": "from_files", "directory": vocab_dir, "oov_token": "<UNK>"})
        vocab = Vocabulary.from_params(params)

        with pytest.raises(AssertionError) as excinfo:
            vocab = Vocabulary.from_params(Params({"type": "from_files", "directory": vocab_dir}))

        assert "OOV token not found!" in str(excinfo.value)

    def test_extend_from_vocab(self):
        vocab1 = Vocabulary(non_padded_namespaces={"1", "2"})
        vocab2 = Vocabulary(non_padded_namespaces={"3"})

        vocab1.add_tokens_to_namespace(["a", "b", "c"], namespace="1")
        vocab1.add_tokens_to_namespace(["d", "e", "f"], namespace="2")

        vocab2.add_tokens_to_namespace(["c", "d", "e"], namespace="1")
        vocab2.add_tokens_to_namespace(["g", "h", "i"], namespace="3")

        vocab1.extend_from_vocab(vocab2)
        assert vocab1.get_namespaces() == {"1", "2", "3"}
        assert vocab1._non_padded_namespaces == {"1", "2", "3"}
        assert vocab1.get_token_to_index_vocabulary("1") == {
            "a": 0,
            "b": 1,
            "c": 2,
            "@@PADDING@@": 3,
            "@@UNKNOWN@@": 4,
            "d": 5,
            "e": 6,
        }
        assert vocab1.get_token_to_index_vocabulary("2") == {
            "d": 0,
            "e": 1,
            "f": 2,
        }
        assert vocab1.get_token_to_index_vocabulary("3") == {
            "g": 0,
            "h": 1,
            "i": 2,
        }
