# pylint: disable=no-self-use,invalid-name, protected-access
import pytest
import torch

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.common.testing import AllenNlpTestCase


class TestBasicTextFieldEmbedder(AllenNlpTestCase):
    def setUp(self):
        super(TestBasicTextFieldEmbedder, self).setUp()
        self.vocab = Vocabulary()
        self.vocab.add_token_to_namespace("1")
        self.vocab.add_token_to_namespace("2")
        self.vocab.add_token_to_namespace("3")
        self.vocab.add_token_to_namespace("4")
        params = Params({
                "token_embedders": {
                        "words1": {
                                "type": "embedding",
                                "embedding_dim": 2
                                },
                        "words2": {
                                "type": "embedding",
                                "embedding_dim": 5
                                },
                        "words3": {
                                "type": "embedding",
                                "embedding_dim": 3
                                }
                        }
                })
        self.token_embedder = BasicTextFieldEmbedder.from_params(vocab=self.vocab, params=params)
        self.inputs = {
                "words1": torch.LongTensor([[0, 2, 3, 5]]),
                "words2": torch.LongTensor([[1, 4, 3, 2]]),
                "words3": torch.LongTensor([[1, 5, 1, 2]])
                }

    def test_get_output_dim_aggregates_dimension_from_each_embedding(self):
        assert self.token_embedder.get_output_dim() == 10

    def test_forward_asserts_input_field_match(self):
        # Total mismatch
        self.inputs['words4'] = self.inputs['words3']
        del self.inputs['words3']
        with pytest.raises(ConfigurationError) as exc:
            self.token_embedder(self.inputs)
        assert exc.match("Mismatched token keys")

        self.inputs['words3'] = self.inputs['words4']

        # Text field has too many inputs
        with pytest.raises(ConfigurationError) as exc:
            self.token_embedder(self.inputs)
        assert exc.match("is generating more keys")

        del self.inputs['words4']


    def test_forward_concats_resultant_embeddings(self):
        assert self.token_embedder(self.inputs).size() == (1, 4, 10)

    def test_forward_works_on_higher_order_input(self):
        params = Params({
                "token_embedders": {
                        "words": {
                                "type": "embedding",
                                "num_embeddings": 20,
                                "embedding_dim": 2,
                                },
                        "characters": {
                                "type": "character_encoding",
                                "embedding": {
                                        "embedding_dim": 4,
                                        "num_embeddings": 15,
                                        },
                                "encoder": {
                                        "type": "cnn",
                                        "embedding_dim": 4,
                                        "num_filters": 10,
                                        "ngram_filter_sizes": [3],
                                        },
                                }
                        }
                })
        token_embedder = BasicTextFieldEmbedder.from_params(vocab=self.vocab, params=params)
        inputs = {
                'words': (torch.rand(3, 4, 5, 6) * 20).long(),
                'characters': (torch.rand(3, 4, 5, 6, 7) * 15).long(),
                }
        assert token_embedder(inputs, num_wrapping_dims=2).size() == (3, 4, 5, 6, 12)

    def test_forward_runs_with_non_bijective_mapping(self):
        elmo_fixtures_path = self.FIXTURES_ROOT / 'elmo'
        options_file = str(elmo_fixtures_path / 'options.json')
        weight_file = str(elmo_fixtures_path / 'lm_weights.hdf5')
        params = Params({
                "token_embedders": {
                        "words": {
                                "type": "embedding",
                                "num_embeddings": 20,
                                "embedding_dim": 2,
                                },
                        "elmo": {
                                "type": "elmo_token_embedder",
                                "options_file": options_file,
                                "weight_file": weight_file
                                },
                        },
                "embedder_to_indexer_map": {"words": ["words"], "elmo": ["elmo", "words"]}
                })
        token_embedder = BasicTextFieldEmbedder.from_params(self.vocab, params)
        inputs = {
                'words': (torch.rand(3, 6) * 20).long(),
                'elmo': (torch.rand(3, 6, 50) * 15).long(),
                }
        token_embedder(inputs)

    def test_old_from_params_new_from_params(self):
        old_params = Params({
                "words1": {
                        "type": "embedding",
                        "embedding_dim": 2
                        },
                "words2": {
                        "type": "embedding",
                        "embedding_dim": 5
                        },
                "words3": {
                        "type": "embedding",
                        "embedding_dim": 3
                        }
                })

        # Allow loading the parameters in the old format
        with pytest.warns(DeprecationWarning):
            old_embedder = BasicTextFieldEmbedder.from_params(params=old_params, vocab=self.vocab)

        new_params = Params({
                "token_embedders": {
                        "words1": {
                                "type": "embedding",
                                "embedding_dim": 2
                                },
                        "words2": {
                                "type": "embedding",
                                "embedding_dim": 5
                                },
                        "words3": {
                                "type": "embedding",
                                "embedding_dim": 3
                                }
                        }
                })

        # But also allow loading the parameters in the new format
        new_embedder = BasicTextFieldEmbedder.from_params(params=new_params, vocab=self.vocab)
        assert old_embedder._token_embedders.keys() == new_embedder._token_embedders.keys()

        assert new_embedder(self.inputs).size() == (1, 4, 10)

    def test_extension_by_vocab(self):
        text_embedder = self.token_embedder
        vocab = self.vocab

        original_token_embedder_weight_words1 = text_embedder.token_embedder_words1.weight
        original_token_embedder_weight_words2 = text_embedder.token_embedder_words2.weight
        original_token_embedder_weight_words3 = text_embedder.token_embedder_words3.weight

        assert tuple(text_embedder.token_embedder_words1.weight.shape) == (6, 2)
        assert tuple(text_embedder.token_embedder_words2.weight.shape) == (6, 5)
        assert tuple(text_embedder.token_embedder_words3.weight.shape) == (6, 3)

        extended_inputs = {
                "words1": torch.LongTensor([[6]]),
                "words2": torch.LongTensor([[7]]),
                "words3": torch.LongTensor([[8]])
                }

        # This should give error for now.
        with pytest.raises(Exception) as _:
            text_embedder(extended_inputs)

        counter = {"tokens": {"5": 1, "6": 1, "7": 1}}
        vocab._extend(counter)

        text_embedder.extend_vocab(vocab)

        assert tuple(text_embedder.token_embedder_words1.weight.shape) == (9, 2)
        assert tuple(text_embedder.token_embedder_words2.weight.shape) == (9, 5)
        assert tuple(text_embedder.token_embedder_words3.weight.shape) == (9, 3)

        # This shouldn't give error now.
        text_embedder(extended_inputs)

        assert torch.all(text_embedder.token_embedder_words1.weight[:6, :]
                         == original_token_embedder_weight_words1[:6, :])
        assert torch.all(text_embedder.token_embedder_words2.weight[:6, :]
                         == original_token_embedder_weight_words2[:6, :])
        assert torch.all(text_embedder.token_embedder_words3.weight[:6, :]
                         == original_token_embedder_weight_words3[:6, :])
