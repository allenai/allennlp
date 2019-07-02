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
        """
        Helper function for the tests below. Finds the weight and training state files in
        self.TEST_DIR, parses their names for the epochs that were saved, deletes them,
        and returns the saved epochs as two lists of integers.
        """
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
        Tests that the default behavior keeps just the last 20 checkpoints.
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

    def test_keep_zero(self):
        checkpointer = Checkpointer(serialization_dir=self.TEST_DIR,
                                    num_serialized_models_to_keep=0)
        for e in range(10):
            checkpointer.save_checkpoint(epoch=e,
                                         model_state={"epoch": e},
                                         training_states={"epoch": e},
                                         is_best_so_far=True)
        files = os.listdir(self.TEST_DIR)
        assert 'model_state_epoch_1.th' not in files
        assert 'training_state_epoch_1.th' not in files

    def test_with_time(self):
        """
        Tests that keep_serialized_model_every_num_seconds parameter causes a checkpoint to be saved
        after enough time has elapsed between epochs.
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
        """
        Users should initialize Trainer either with an instance of Checkpointer or by specifying
        parameter values for num_serialized_models_to_keep and keep_serialized_model_every_num_seconds.
        Check that Trainer raises a ConfigurationError if both methods are used at the same time.
        """
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
        Tests that registering Checkpointer subclasses works correctly.
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
