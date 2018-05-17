from allennlp.data.iterators import EpochTrackingBucketIterator
from allennlp.tests.data.iterators.basic_iterator_test import IteratorTest


class EpochTrackingBucketIteratorTest(IteratorTest):
    def setUp(self):
        # The super class creates a self.instances field and populates it with some instances with
        # TextFields.
        super(EpochTrackingBucketIteratorTest, self).setUp()
        self.iterator = EpochTrackingBucketIterator(sorting_keys=[["text", "num_tokens"]])
        # We'll add more to create a second dataset.
        self.more_instances = [
                self.create_instance(["this", "is", "a", "sentence"]),
                self.create_instance(["this", "is", "in", "the", "second", "dataset"]),
                self.create_instance(["so", "is", "this", "one"])
                ]

    def test_iterator_tracks_epochs_per_dataset(self):
        generated_dataset1 = list(self.iterator(self.instances, num_epochs=2))
        generated_dataset2 = list(self.iterator(self.more_instances, num_epochs=2))

        # First dataset has five sentences. See ``IteratorTest.setUp``
        assert generated_dataset1[0]["epoch_num"] == [0, 0, 0, 0, 0]
        assert generated_dataset1[1]["epoch_num"] == [1, 1, 1, 1, 1]
        # Second dataset has three sentences.
        assert generated_dataset2[0]["epoch_num"] == [0, 0, 0]
        assert generated_dataset2[1]["epoch_num"] == [1, 1, 1]
