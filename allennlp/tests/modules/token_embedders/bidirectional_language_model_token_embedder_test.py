from .language_model_token_embedder_test import TestLanguageModelTokenEmbedder


class TestBidirectionalLanguageModelTokenEmbedder(TestLanguageModelTokenEmbedder):
    def setUp(self):
        super().setUp()
        self.set_up_model(
            self.FIXTURES_ROOT
            / "language_model"
            / "bidirectional_lm_characters_token_embedder.jsonnet",
            self.FIXTURES_ROOT / "data" / "conll2003.txt",
        )


class TestBidirectionalLanguageModelTokenEmbedderWithoutBosEos(TestLanguageModelTokenEmbedder):
    def setUp(self):
        super().setUp()
        self.set_up_model(
            self.FIXTURES_ROOT
            / "language_model"
            / "bidirectional_lm_characters_token_embedder_without_bos_eos.jsonnet",
            self.FIXTURES_ROOT / "data" / "conll2003.txt",
        )
