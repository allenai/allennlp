# pylint: disable=no-self-use,invalid-name
import pytest
import spacy
import torch
import numpy
import h5py

from allennlp.common.testing import ModelTestCase, AllenNlpTestCase
from allennlp.common.params import Params
from allennlp.data.dataset import Batch
from allennlp.data import Token
from allennlp.data.token_indexers import OpenaiTransformerBytePairIndexer
from allennlp.data.token_indexers.openai_transformer_byte_pair_indexer import text_standardize
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.openai_transformer import OpenaiTransformer
from allennlp.modules.token_embedders import OpenaiTransformerEmbedder
from allennlp.nn.util import get_range_vector


# Skip this one, it's an expensive test.
@pytest.mark.skip()
class TestOpenaiTransformerEmbedderLarge(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(self.FIXTURES_ROOT / 'openai_transformer' / 'config_large.jsonnet',
                          self.FIXTURES_ROOT / 'data' / 'conll2003.txt')

    def test_tagger_with_openai_token_embedder_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_tagger_with_openai_token_embedder_forward_pass_runs_correctly(self):
        dataset = Batch(self.instances)
        dataset.index_instances(self.vocab)
        training_tensors = dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        tags = output_dict['tags']
        assert len(tags) == 2
        assert len(tags[0]) == 7
        assert len(tags[1]) == 7
        for example_tags in tags:
            for tag_id in example_tags:
                tag = self.model.vocab.get_token_from_index(tag_id, namespace="labels")
                assert tag in {'O', 'I-ORG', 'I-PER', 'I-LOC'}


class TestOpenaiTransformerEmbedderSmall(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(self.FIXTURES_ROOT / 'openai_transformer' / 'config_small.jsonnet',
                          self.FIXTURES_ROOT / 'data' / 'conll2003.txt')

    def test_tagger_with_openai_token_embedder_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def _get_training_tensors(self):
        dataset = Batch(self.instances)
        dataset.index_instances(self.vocab)
        return dataset.as_tensor_dict()

    def test_tagger_with_openai_token_embedder_forward_pass_runs_correctly(self):
        training_tensors = self._get_training_tensors()
        output_dict = self.model(**training_tensors)
        tags = output_dict['tags']
        assert len(tags) == 2
        assert len(tags[0]) == 7
        assert len(tags[1]) == 7
        for example_tags in tags:
            for tag_id in example_tags:
                tag = self.model.vocab.get_token_from_index(tag_id, namespace="labels")
                assert tag in {'O', 'I-ORG', 'I-PER', 'I-LOC'}

    def test_openai_can_run_with_top_layer(self):
        params = Params({
                "transformer": {
                        "model_path": "allennlp/tests/fixtures/openai_transformer/transformer_small.tar.gz",
                        "embedding_dim": 10,
                        "num_heads": 2,
                        "num_layers": 2,
                        "vocab_size": 50,
                        "n_ctx": 50
                },
                "top_layer_only": True
        })
        embedder = OpenaiTransformerEmbedder.from_params(params)
        training_tensors = self._get_training_tensors()
        output = embedder(training_tensors['tokens']['openai_transformer'],
                          training_tensors['tokens']['openai_transformer-offsets'])
        assert list(output.shape) == [2, 7, 10]

    def test_openai_can_run_with_no_offsets(self):
        params = Params({
                "transformer": {
                        "model_path": "allennlp/tests/fixtures/openai_transformer/transformer_small.tar.gz",
                        "embedding_dim": 10,
                        "num_heads": 2,
                        "num_layers": 2,
                        "vocab_size": 50,
                        "n_ctx": 50
                },
        })
        embedder = OpenaiTransformerEmbedder.from_params(params)
        training_tensors = self._get_training_tensors()
        output = embedder(training_tensors['tokens']['openai_transformer'])
        assert list(output.shape) == [2, 2, 10]


# Skip this one, it's an expensive test.
@pytest.mark.skip()
class TestOpenAiTransformerEmbedderCorrectWithFixture(AllenNlpTestCase):
    """
    Test that the implementation produces same embeddings as tensorflow model
    """
    def test_openai_transformer_matches_tensorflow(self):
        model_path = "https://allennlp.s3.amazonaws.com/models/openai-transformer-lm-2018.07.23.tar.gz"
        indexer = OpenaiTransformerBytePairIndexer(model_path=model_path)
        transformer = OpenaiTransformer(model_path=model_path)

        # get the test sentences
        with open(self.FIXTURES_ROOT / 'openai_transformer' / 'text.txt', 'r') as fin:
            sentences = fin.read().strip().split('\n')

        # tokenize and check that indices are correct
        nlp = spacy.load('en_core_web_sm')

        # make a batch of two sentences
        batch_indices = []
        batch_lengths = []
        for k, sentence in enumerate(sentences):
            tokens = [token.text for token in nlp(text_standardize(sentence)) if not token.is_space]
            indices = indexer.tokens_to_indices(
                    [Token(token) for token in tokens], Vocabulary(), 'openai_indexer'
            )
            batch_indices.append(indices['openai_indexer'])
            batch_lengths.append(len([i for i in indices['openai_indexer'] if i != 0]))
        batch_indices = torch.from_numpy(numpy.array(batch_indices))
        batch_size, num_timesteps = batch_indices.size()
        vocab_size = transformer.vocab_size - transformer.n_ctx
        positional_encodings = get_range_vector(num_timesteps, device=-1) + vocab_size

        # Combine the inputs with positional encodings
        batch_tensor = torch.stack([
                batch_indices,   # (batch_size, num_timesteps)
                positional_encodings.expand(batch_size, num_timesteps)
        ], dim=-1)

        # run the LM
        transformer.eval()
        activations = transformer(batch_tensor)

        # load the expected activations
        expected_activations = []
        with h5py.File(self.FIXTURES_ROOT / 'openai_transformer' / 'expected_embeddings.hdf5', 'r') as fin:
            expected_activations.append(fin['0'][...])
            expected_activations.append(fin['1'][...])

        # just check the top layer
        for k in range(2):
            actual = activations[-1][k, :batch_lengths[k], :].numpy()
            expected = expected_activations[k]
            numpy.testing.assert_almost_equal(expected, actual, decimal=5)


def create_small_test_fixture(output_dir: str = '/tmp') -> None:
    """
    This is how I created the transformer_model.tar.gz.
    After running this, go to the specified output dir and run

        tar -czvf transformer_model.tar.gz model/

    In case you need to regenerate the fixture for some reason.
    """
    import json
    import pathlib

    model_dir = pathlib.Path(output_dir) / 'model'
    model_dir.mkdir(exist_ok=True)  # pylint: disable=no-member

    symbols = ["e", "w", "o", "wo", "."]
    byte_pairs = [(sym1, sym2 + end)
                  for sym1 in symbols        # prefer earlier first symbol
                  for sym2 in symbols        # if tie, prefer earlier second symbol
                  for end in ('</w>', '')]   # if tie, prefer ending a word
    encoding = {f"{sym1}{sym2}": idx for idx, (sym1, sym2) in enumerate(byte_pairs)}
    encoding["<unk>"] = 0

    with open(model_dir / 'encoder_bpe.json', 'w') as encoder_file:
        json.dump(encoding, encoder_file)

    with open(model_dir / 'vocab.bpe', 'w') as bpe_file:
        bpe_file.write("#version 0.0\n")
        for sym1, sym2 in byte_pairs:
            bpe_file.write(f"{sym1} {sym2}\n")
        bpe_file.write("\n")

    transformer = OpenaiTransformer(embedding_dim=10, num_heads=2, num_layers=2, vocab_size=(50 + 50), n_ctx=50)
    transformer.dump_weights(output_dir, num_pieces=2)
