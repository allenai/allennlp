from allennlp.common.testing.model_test_case import ModelTestCase


class GraphParserTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(
            self.FIXTURES_ROOT / "graph_parser" / "experiment.json",
            self.FIXTURES_ROOT / "data" / "dm.sdp",
        )

    def test_graph_parser_can_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()

    def test_model_can_decode(self):
        self.model.eval()
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        decode_output_dict = self.model.decode(output_dict)

        assert set(decode_output_dict.keys()) == {
            "arc_loss",
            "tag_loss",
            "loss",
            "arcs",
            "arc_tags",
            "arc_tag_probs",
            "arc_probs",
            "tokens",
            "mask",
        }
