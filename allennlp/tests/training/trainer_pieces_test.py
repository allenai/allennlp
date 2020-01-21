from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Instance, Token, Vocabulary
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.training.trainer_pieces import TrainerPieces


class TestTrainerPieces(AllenNlpTestCase):
    def test_create_vocab(self):
        params = Params({})

        token_indexer = SingleIdTokenIndexer()
        instance1 = Instance(
            fields={
                "tokens": TextField(
                    [Token("This"), Token("is"), Token("a"), Token("test"), Token(".")],
                    {"tokens": token_indexer},
                ),
                "label": LabelField("T"),
            }
        )
        instance2 = Instance(
            fields={
                "tokens": TextField(
                    [Token("This"), Token("is"), Token("another"), Token("test"), Token(".")],
                    {"tokens": token_indexer},
                ),
                "label": LabelField("F"),
            }
        )
        datasets = {"train": [instance1, instance2]}

        vocabulary_params = Params({})
        vocabulary_path = "/tmp/path_should_not_be_used"

        vocab = TrainerPieces.create_or_extend_vocab(
            params=params,
            datasets=datasets,
            vocabulary_params=vocabulary_params,
            vocabulary_path=vocabulary_path,
            vocab=None,
            recover=False,
        )

        # Additional 2 tokens are '@@PADDING@@' and '@@UNKNOWN@@' by default.
        self.assertEqual(
            vocab.get_vocab_size("tokens"), len({"This", "is", "a", "another", "test", "."}) + 2
        )
        self.assertEqual(vocab.get_vocab_size("labels"), len({"T", "F"}))

    def test_extend_vocab(self):
        vocab = Vocabulary(tokens_to_add={"tokens": ["This", "extend"]})

        params = Params({})

        token_indexer = SingleIdTokenIndexer()
        instance1 = Instance(
            fields={
                "tokens": TextField(
                    [Token("This"), Token("is"), Token("a"), Token("test"), Token(".")],
                    {"tokens": token_indexer},
                ),
                "label": LabelField("T"),
            }
        )
        instance2 = Instance(
            fields={
                "tokens": TextField(
                    [Token("This"), Token("is"), Token("another"), Token("test"), Token(".")],
                    {"tokens": token_indexer},
                ),
                "label": LabelField("F"),
            }
        )
        datasets = {"train": [instance1, instance2]}

        vocabulary_params = Params({})
        vocabulary_path = "/tmp/path_should_not_be_used"

        vocab = TrainerPieces.create_or_extend_vocab(
            params=params,
            datasets=datasets,
            vocabulary_params=vocabulary_params,
            vocabulary_path=vocabulary_path,
            vocab=vocab,
            recover=False,
        )

        # Additional 2 tokens are '@@PADDING@@' and '@@UNKNOWN@@' by default.
        self.assertEqual(
            vocab.get_vocab_size("tokens"),
            len({"This", "is", "a", "another", "test", "extend", "."}) + 2,
        )
        self.assertEqual(vocab.get_vocab_size("labels"), len({"T", "F"}))
