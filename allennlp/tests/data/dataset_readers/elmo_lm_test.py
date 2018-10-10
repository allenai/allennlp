import os
from typing import cast

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.fields import TextField

from allennlp.data.dataset_readers.elmo_lm import LMDatasetReader


class TestLMDatasetReader(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.FIXTURES = self.FIXTURES_ROOT / "elmo_port"

    def test_lm_dataset_text_to_instance(self):
        dataset = LMDatasetReader()

        instance = dataset.text_to_instance('The only sentence.')
        text = [t.text for t in cast(TextField, instance.fields["source"]).tokens]
        self.assertEqual(text, ["The", "only", "sentence", "."])

    def test_lm_dataset_read(self):
        prefix = os.path.join(self.FIXTURES, 'single_sentence.txt')
        dataset = LMDatasetReader()
        with open(prefix, 'r') as fin:
            sentence = fin.read().strip()
        expected_batch = dataset.text_to_instance(sentence)
        for batch in dataset.read(prefix):
            break
        self.assertEqual(sorted(list(expected_batch.fields.keys())),
                         sorted(list(batch.fields.keys())))
        for k in expected_batch.fields.keys():
            self.assertTrue(str(expected_batch.fields[k]) == str(batch.fields[k]))

    def test_lm_dataset_read_shards(self):
        prefix = os.path.join(self.FIXTURES, 'shards/*')
        dataset = LMDatasetReader(loop_indefinitely=False)
        for k, _ in enumerate(dataset.read(prefix)):
            pass
        self.assertEqual(k, 999)

    def test_lm_dataset_read_shards_max_sequence_length(self):
        prefix = os.path.join(self.FIXTURES, 'shards/*')
        dataset = LMDatasetReader(loop_indefinitely=False, max_sequence_length=10)
        for k, _ in enumerate(dataset.read(prefix)):
            pass
        self.assertEqual(k, 148)

    def test_lm_dataset_read_shards_loops_indefinitely(self):
        prefix = os.path.join(self.FIXTURES, 'shards/*')
        dataset = LMDatasetReader(loop_indefinitely=True)
        for k, _ in enumerate(dataset.read(prefix)):
            if k == 1000:
                break
        # There are 1000 instances in the fixture shards. We'd like to verify we loop at least once.
        self.assertTrue(k > 999)
