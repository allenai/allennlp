
import os
from typing import cast

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Instance
from allennlp.data.fields import TextField

from allennlp.data.dataset_readers.elmo_lm import LMDatasetReader

def get_text(key: str, instance: Instance):
    return [t.text for t in cast(TextField, instance.fields[key]).tokens]

class TestLMDatasetReader(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.FIXTURES = self.FIXTURES_ROOT / "elmo_port"

    def test_lm_dataset_text_to_instance(self):
        dataset = LMDatasetReader()

        instance = dataset.text_to_instance('The only sentence')
        self.assertTrue(
            get_text('characters', instance) == ["@start@", "The", "only", "sentence", "@end@"]
        )
        self.assertTrue(get_text('tokens', instance) == ["@start@", "The", "only", "sentence", "@end@"])
        self.assertTrue(
            get_text('forward_targets', instance) == ["The", "only", "sentence", "@end@"]
        )
        self.assertTrue(
            get_text('backward_targets', instance) == ["@@PADDING@@", "@start@", "The", "only", "sentence"]
        )

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
        dataset = LMDatasetReader()
        for k, _ in enumerate(dataset.read(prefix)):
            if k == 20:
                break
        self.assertTrue(k == 20)

    def test_lm_dataset_read_shards_test(self):
        prefix = os.path.join(self.FIXTURES, 'shards/*')
        dataset = LMDatasetReader(test=True)
        for _ in dataset.read(prefix):
            pass
        self.assertTrue(True)
