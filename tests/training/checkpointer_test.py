import os
import time
from typing import Optional

from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.params import Params
from allennlp.training import Checkpointer, Trainer
from allennlp.training.trainer import TrainerCheckpoint


class FakeTrainer(Trainer):
    def __init__(self, model_state, training_state):
        self._model_state = model_state
        self._training_state = training_state

    def get_checkpoint_state(self) -> TrainerCheckpoint:
        return TrainerCheckpoint(self._model_state, self._training_state)


class TestCheckpointer(AllenNlpTestCase):
    def retrieve_and_delete_saved(self, shard: Optional[int] = None):
        """
        Helper function for the tests below. Finds the weight and training state files in
        self.TEST_DIR, parses their names for the epochs that were saved, deletes them,
        and returns the saved epochs as two lists of integers.
        """
        serialization_files = os.listdir(self.TEST_DIR)

        model_checkpoints = [x for x in serialization_files if "model_state_" in x]
        if shard is not None:
            model_checkpoints = [x for x in model_checkpoints if x.endswith(f"_w{shard}.th")]
        found_model_states = [Checkpointer._parse_model_state_path(x) for x in model_checkpoints]
        for f in model_checkpoints:
            os.remove(os.path.join(self.TEST_DIR, f))

        training_checkpoints = [x for x in serialization_files if "training_state_" in x]
        if shard is not None:
            training_checkpoints = [x for x in training_checkpoints if x.endswith(f"_w{shard}.th")]
        found_training_states = [
            Checkpointer._parse_training_state_path(x) for x in training_checkpoints
        ]
        for f in training_checkpoints:
            os.remove(os.path.join(self.TEST_DIR, f))
        return sorted(found_model_states), sorted(found_training_states)

    def test_default(self):
        """
        Tests that the default behavior keeps just the last 2 checkpoints.
        """
        default_num_to_keep = 2
        num_epochs = 5
        target = [(e, 0) for e in range(num_epochs - default_num_to_keep, num_epochs)]

        checkpointer = Checkpointer(serialization_dir=self.TEST_DIR)
        for epochs_completed in range(num_epochs):
            for batches_completed in [0, 5, 10]:
                state = {
                    "epochs_completed": epochs_completed,
                    "batches_in_epoch_completed": batches_completed,
                }
                checkpointer.maybe_save_checkpoint(
                    FakeTrainer(model_state=state, training_state=state),
                    epochs_completed,
                    batches_completed,
                )
        models, training = self.retrieve_and_delete_saved()
        assert models == training == target

    def test_keep_zero(self):
        checkpointer = Checkpointer(serialization_dir=self.TEST_DIR, keep_most_recent_by_count=0)
        for epochs_completed in range(5):
            state = {"epochs_completed": epochs_completed, "batches_in_epoch_completed": 0}
            checkpointer.maybe_save_checkpoint(
                FakeTrainer(model_state=state, training_state=state), epochs_completed, 0
            )
        files = os.listdir(self.TEST_DIR)
        assert not any("model_state_" in x for x in files)
        assert not any("training_state_" in x for x in files)

    def test_with_time(self):
        num_epochs = 30
        pauses = [5, 18, 26]
        target = [(e, 0) for e in pauses]
        checkpointer = Checkpointer(
            serialization_dir=self.TEST_DIR,
            save_completed_epochs=False,
            save_every_num_seconds=1,
            keep_most_recent_by_count=3,
        )
        for e in range(num_epochs):
            if e in pauses:
                time.sleep(2)
            state = {"epochs_completed": e, "batches_in_epoch_completed": 0}
            checkpointer.maybe_save_checkpoint(
                trainer=FakeTrainer(model_state=state, training_state=state),
                num_epochs_completed=e,
                num_batches_in_epoch_completed=0,
            )
        models, training = self.retrieve_and_delete_saved()
        assert models == training == target

    def test_registered_subclass(self):
        """
        Tests that registering Checkpointer subclasses works correctly.
        """

        serialization_dir = str(self.TEST_DIR)

        @Checkpointer.register("checkpointer_subclass")
        class CheckpointerSubclass(Checkpointer):
            def __init__(self, x: int, y: int) -> None:
                super().__init__(serialization_dir)
                self.x = x
                self.y = y

        sub_inst = Checkpointer.from_params(
            Params({"type": "checkpointer_subclass", "x": 1, "y": 3})
        )
        assert sub_inst.__class__ == CheckpointerSubclass
        assert sub_inst.x == 1 and sub_inst.y == 3

    def test_base_class_from_params(self):
        Checkpointer.from_params(Params({}), serialization_dir=self.TEST_DIR)

    def test_default_distributed_with_sharded_state(self):
        """
        Simulates using the Checkpointer during distributed training with a sharded model.
        """
        world_size = 2
        default_num_to_keep = 2
        num_epochs = 5
        target = [(e, 0) for e in range(num_epochs - default_num_to_keep, num_epochs)]

        checkpointers = [Checkpointer(serialization_dir=self.TEST_DIR) for _ in range(world_size)]
        for i, checkpointer in enumerate(checkpointers):
            checkpointer._rank = i
            checkpointer.state_is_sharded = True

        for epochs_completed in range(num_epochs):
            for batches_completed in [0, 5, 10]:
                for i, checkpointer in enumerate(checkpointers):
                    state = {
                        "epochs_completed": epochs_completed,
                        "batches_in_epoch_completed": batches_completed,
                        "rank": i,
                    }
                    checkpointer.maybe_save_checkpoint(
                        FakeTrainer(model_state=state, training_state=state),
                        epochs_completed,
                        batches_completed,
                    )

        for i, checkpointer in enumerate(checkpointers):
            checkpoint = checkpointer.load_checkpoint()
            assert checkpoint is not None
            model_state, training_state = checkpoint
            assert model_state["rank"] == i
            assert training_state["rank"] == i

            models, training = self.retrieve_and_delete_saved(shard=i)
            assert models == training == target
