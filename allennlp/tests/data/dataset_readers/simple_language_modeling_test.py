import os
from typing import cast

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.fields import TextField

from allennlp.data.dataset_readers import SimpleLanguageModelingDatasetReader


class TestSimpleLanguageModelingDatasetReader(AllenNlpTestCase):
    FIXTURES = AllenNlpTestCase.FIXTURES_ROOT / "language_modeling"

    def test_text_to_instance(self):
        dataset = SimpleLanguageModelingDatasetReader(start_tokens=["<S>"], end_tokens=["</S>"])

        instance = dataset.text_to_instance("The only sentence.")
        text = [t.text for t in cast(TextField, instance.fields["source"]).tokens]
        self.assertEqual(text, ["<S>", "The", "only", "sentence", ".", "</S>"])

    def test_read_single_sentence(self):
        prefix = os.path.join(self.FIXTURES, "single_sentence.txt")
        dataset = SimpleLanguageModelingDatasetReader()
        with open(prefix, "r") as fin:
            sentence = fin.read().strip()
        expected_batch = dataset.text_to_instance(sentence)
        batch = None
        for batch in dataset.read(prefix):
            break
        self.assertEqual(
            sorted(list(expected_batch.fields.keys())), sorted(list(batch.fields.keys()))
        )
        for k in expected_batch.fields.keys():
            self.assertTrue(str(expected_batch.fields[k]) == str(batch.fields[k]))

    def test_read_multiple_sentences(self):
        prefix = os.path.join(self.FIXTURES, "shards/shard0")
        dataset = SimpleLanguageModelingDatasetReader()
        k = -1
        for k, _ in enumerate(dataset.read(prefix)):
            pass
        self.assertEqual(k, 99)

    def test_max_sequence_length(self):
        prefix = os.path.join(self.FIXTURES, "shards/shard0")
        dataset = SimpleLanguageModelingDatasetReader(
            max_sequence_length=10, start_tokens=["<S>"], end_tokens=["</S>"]
        )
        k = -1
        for k, _ in enumerate(dataset.read(prefix)):
            pass
        self.assertEqual(k, 7)
