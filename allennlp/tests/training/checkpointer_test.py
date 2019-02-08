# pylint: disable=invalid-name
import os
import re
import time

from allennlp.common.testing import AllenNlpTestCase
from allennlp.training.checkpointer import Checkpointer
from allennlp.common.params import Params
from allennlp.training.trainer import Trainer
from allennlp.common.checks import ConfigurationError


class TestCheckpointer(AllenNlpTestCase):
    def retrieve_and_delete_saved(self):
        serialization_files = os.listdir(self.TEST_DIR)
        model_checkpoints = [x for x in serialization_files if "model_state_epoch" in x]
        found_model_epochs = [int(re.search(r"model_state_epoch_([0-9\.\-]+)\.th", x).group(1))
                              for x in model_checkpoints]
        for f in model_checkpoints:
            os.remove(os.path.join(self.TEST_DIR, f))
        training_checkpoints = [x for x in serialization_files if "training_state_epoch" in x]
        found_training_epochs = [int(re.search(r"training_state_epoch_([0-9\.\-]+)\.th", x).group(1))
                                 for x in training_checkpoints]
        for f in training_checkpoints:
            os.remove(os.path.join(self.TEST_DIR, f))
        return sorted(found_model_epochs), sorted(found_training_epochs)

    def test_default(self):
        """
        Tests against Trainer's default values for num_to_keep
        """
        default_num_to_keep = 20
        num_epochs = 30
        target = list(range(num_epochs - default_num_to_keep, num_epochs))

        checkpointer = Checkpointer(serialization_dir=self.TEST_DIR)

        for e in range(num_epochs):
            checkpointer.save_checkpoint(epoch=e,
                                         model_state={"epoch": e},
                                         training_states={"epoch": e},
                                         is_best_so_far=False)
        models, training = self.retrieve_and_delete_saved()
        assert models == training == target

    def test_with_time(self):
        """
        Tests consistent behavior for keep_serialized_model_every_num_seconds
        """
        num_to_keep = 10
        num_epochs = 30
        target = list(range(num_epochs - num_to_keep, num_epochs))
        pauses = [5, 18, 26]
        target = sorted(set(target + pauses))
        checkpointer = Checkpointer(serialization_dir=self.TEST_DIR,
                                    num_serialized_models_to_keep=num_to_keep,
                                    keep_serialized_model_every_num_seconds=1)
        for e in range(num_epochs):
            if e in pauses:
                time.sleep(2)
            checkpointer.save_checkpoint(epoch=e,
                                         model_state={"epoch": e},
                                         training_states={"epoch": e},
                                         is_best_so_far=False)
        models, training = self.retrieve_and_delete_saved()
        assert models == training == target

    def test_configuration_error_when_passed_as_conflicting_argument_to_trainer(self):
        with self.assertRaises(ConfigurationError):
            Trainer(None, None, None, None,
                    num_serialized_models_to_keep=30,
                    keep_serialized_model_every_num_seconds=None,
                    checkpointer=Checkpointer(serialization_dir=self.TEST_DIR,
                                              num_serialized_models_to_keep=40,
                                              keep_serialized_model_every_num_seconds=2))
        with self.assertRaises(ConfigurationError):
            Trainer(None, None, None, None,
                    num_serialized_models_to_keep=20,
                    keep_serialized_model_every_num_seconds=2,
                    checkpointer=Checkpointer(serialization_dir=self.TEST_DIR,
                                              num_serialized_models_to_keep=40,
                                              keep_serialized_model_every_num_seconds=2))
        try:
            Trainer(None, None, None, None,
                    checkpointer=Checkpointer(serialization_dir=self.TEST_DIR,
                                              num_serialized_models_to_keep=40,
                                              keep_serialized_model_every_num_seconds=2))
        except ConfigurationError:
            self.fail("Configuration Error raised for passed checkpointer")

    def test_registered_subclass(self):
        """
        Tests that registering subclasses works correctly.
        """

        @Checkpointer.register("checkpointer_subclass")
        class CheckpointerSubclass(Checkpointer):
            def __init__(self, x: int, y: int) -> None:
                super().__init__()
                self.x = x
                self.y = y

        sub_inst = Checkpointer.from_params(Params({"type": "checkpointer_subclass", "x": 1, "y": 3}))
        assert sub_inst.__class__ == CheckpointerSubclass
        assert sub_inst.x == 1 and sub_inst.y == 3
