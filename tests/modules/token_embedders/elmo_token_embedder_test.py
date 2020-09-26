import torch

from allennlp.common import Params
from allennlp.common.testing import ModelTestCase
from allennlp.data.batch import Batch
from allennlp.modules.token_embedders import ElmoTokenEmbedder


class TestElmoTokenEmbedder(ModelTestCase):
    def setup_method(self):
        super().setup_method()
        self.set_up_model(
            self.FIXTURES_ROOT / "elmo" / "config" / "characters_token_embedder.json",
            self.FIXTURES_ROOT / "data" / "conll2003.txt",
        )

    def test_tagger_with_elmo_token_embedder_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_tagger_with_elmo_token_embedder_forward_pass_runs_correctly(self):
        dataset = Batch(self.instances)
        dataset.index_instances(self.vocab)
        training_tensors = dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        probs = output_dict["class_probabilities"]
        assert probs.size() == (2, 7, self.model.vocab.get_vocab_size("labels"))

    def test_forward_works_with_projection_layer(self):
        params = Params(
            {
                "options_file": self.FIXTURES_ROOT / "elmo" / "options.json",
                "weight_file": self.FIXTURES_ROOT / "elmo" / "lm_weights.hdf5",
                "projection_dim": 20,
            }
        )
        word1 = [0] * 50
        word2 = [0] * 50
        word1[0] = 6
        word1[1] = 5
        word1[2] = 4
        word1[3] = 3
        word2[0] = 3
        word2[1] = 2
        word2[2] = 1
        word2[3] = 0
        embedding_layer = ElmoTokenEmbedder.from_params(vocab=None, params=params)
        assert embedding_layer.get_output_dim() == 20

        input_tensor = torch.LongTensor([[word1, word2]])
        embedded = embedding_layer(input_tensor).data.numpy()
        assert embedded.shape == (1, 2, 20)

        input_tensor = torch.LongTensor([[[word1]]])
        embedded = embedding_layer(input_tensor).data.numpy()
        assert embedded.shape == (1, 1, 1, 20)
