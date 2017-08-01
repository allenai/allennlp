

from allennlp.testing.test_case import AllenNlpTestCase
from allennlp.commands.train import train_model


class TestTrain(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.write_sequence_tagging_data()

    def test_train_model(self):
        trainer_params = {
                "model": {
                        "type": "simple_tagger",
                        "text_field_embedder": {
                                "tokens": {
                                        "type": "embedding",
                                        "embedding_dim": 5
                                }
                        },
                        "stacked_encoder": {
                                "type": "lstm",
                                "input_size": 5,
                                "hidden_size": 7,
                                "num_layers": 2
                        }
                },
                "dataset_reader": {"type": "sequence_tagging"},
                "train_data_path": self.TRAIN_FILE,
                "iterator": {"type": "basic", "batch_size": 2},
                "optimizer": "adam",
                "num_epochs": 2
        }
        train_model(trainer_params)
