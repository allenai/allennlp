import copy
import tempfile

from allennlp.common.testing import ModelTestCase
from allennlp.common.params import Params
from allennlp.data.dataset import Batch
from allennlp.models import Model
from allennlp.modules.scalar_mix import ScalarMix


class TestLanguageModelTokenEmbedder(ModelTestCase):
    def setUp(self):
        super().setUp()
        self._model_fp = self.FIXTURES_ROOT / "language_model" / "characters_token_embedder.json"
        self.set_up_model(self._model_fp, self.FIXTURES_ROOT / "data" / "conll2003.txt")

    def test_tagger_with_language_model_token_embedder_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_tagger_with_language_model_token_embedder_forward_pass_runs_correctly(self):
        dataset = Batch(self.instances)
        dataset.index_instances(self.vocab)
        training_tensors = dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        tags = output_dict["tags"]
        assert len(tags) == 2
        assert len(tags[0]) == 7
        assert len(tags[1]) == 7
        for example_tags in tags:
            for tag_id in example_tags:
                tag = self.model.vocab.get_token_from_index(tag_id, namespace="labels")
                assert tag in {"O", "I-ORG", "I-PER", "I-LOC"}
        # Ensure that the scalar mix model exists within the Language Model
        # TokenEmbedder
        scalar_mix = self.model.text_field_embedder._token_embedders["elmo"]._scalar_mix
        assert isinstance(scalar_mix, ScalarMix)

    def test_tagger_with_language_model_token_embedder_top_only_layer(self):
        params = Params.from_file(self._model_fp).duplicate()
        elmo_params = {
            "type": "language_model_token_embedder",
            "archive_file": "allennlp/tests/fixtures/language_model/model.tar.gz",
            "dropout": 0.2,
            "bos_eos_tokens": ["<S>", "</S>"],
            "remove_bos_eos": True,
            "requires_grad": True,
            "top_layer_only": True,
        }
        params["model"]["text_field_embedder"]["token_embedders"]["elmo"] = elmo_params
        params_copy = copy.deepcopy(params)
        model = Model.from_params(vocab=self.vocab, params=params_copy.get("model"))
        with tempfile.NamedTemporaryFile(mode="w+") as temp_file:
            params.to_file(temp_file.name)
            self.ensure_model_can_train_save_and_load(temp_file.name)
        assert model.text_field_embedder._token_embedders["elmo"]._scalar_mix is None


class TestLanguageModelTokenEmbedderWithoutBosEos(TestLanguageModelTokenEmbedder):
    def setUp(self):
        super().setUp()
        self.set_up_model(
            self.FIXTURES_ROOT
            / "language_model"
            / "characters_token_embedder_without_bos_eos.jsonnet",
            self.FIXTURES_ROOT / "data" / "conll2003.txt",
        )
