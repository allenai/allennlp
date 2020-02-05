import torch

from allennlp.common.testing import ModelTestCase


class CorefTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(
            self.FIXTURES_ROOT / "coref" / "experiment.json",
            self.FIXTURES_ROOT / "coref" / "coref.gold_conll",
        )

    def test_coref_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_coref_bert_model_can_train_save_and_load(self):
        self.set_up_model(
            self.FIXTURES_ROOT / "coref" / "coref_bert_lstm_small.jsonnet",
            self.FIXTURES_ROOT / "coref" / "coref.gold_conll",
        )
        self.ensure_model_can_train_save_and_load(
            self.param_file,
            gradients_to_ignore={
                "_text_field_embedder.token_embedder_tokens._matched_embedder"
                ".transformer_model.pooler.weight",
                "_text_field_embedder.token_embedder_tokens._matched_embedder"
                ".transformer_model.pooler.bias",
            },
        )

    def test_decode(self):

        spans = torch.LongTensor([[1, 2], [3, 4], [3, 7], [5, 6], [14, 56], [17, 80]])

        antecedent_indices = torch.LongTensor(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [2, 1, 0, 0, 0, 0],
                [3, 2, 1, 0, 0, 0],
                [4, 3, 2, 1, 0, 0],
            ]
        )

        spans = spans.unsqueeze(0)
        antecedent_indices = antecedent_indices
        # Indices into `antecedent_indices` indicating the predicted antecedent
        # index in `top_spans`.
        predicted_antecedents = torch.LongTensor([-1, 0, -1, -1, 1, 3])
        predicted_antecedents = predicted_antecedents.unsqueeze(0)
        output_dict = {
            "top_spans": spans,
            "antecedent_indices": antecedent_indices,
            "predicted_antecedents": predicted_antecedents,
        }
        output = self.model.decode(output_dict)

        clusters = output["clusters"][0]
        gold1 = [(1, 2), (3, 4), (17, 80)]
        gold2 = [(3, 7), (14, 56)]

        assert len(clusters) == 2
        assert gold1 in clusters
        assert gold2 in clusters
