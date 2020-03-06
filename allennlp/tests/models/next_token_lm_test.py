from allennlp.common.testing import ModelTestCase

from ..modules.language_model_heads.linear import LinearLanguageModelHead  # noqa: F401


class TestNextTokenLanguageModel(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(
            self.FIXTURES_ROOT / "next_token_lm" / "experiment.json",
            self.FIXTURES_ROOT / "language_model" / "sentences.txt",
        )

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
