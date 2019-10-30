from allennlp.data.dataset_readers import UniversalDependenciesMultiLangDatasetReader
from allennlp.data.iterators.same_language_iterator import (
    SameLanguageIterator,
    SameLanguageIteratorStub,
)
from allennlp.data.vocabulary import Vocabulary
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.dataset import Batch


class SameLanguageIteratorTest(AllenNlpTestCase):
    data_path = AllenNlpTestCase.FIXTURES_ROOT / "data" / "dependencies_multilang" / "*"

    def test_instances_of_different_languages_are_in_different_batches(self):

        for iterator in [
            SameLanguageIterator(batch_size=2, sorting_keys=[["words", "num_tokens"]]),
            SameLanguageIteratorStub(batch_size=2, sorting_keys=[["words", "num_tokens"]]),
        ]:

            reader = UniversalDependenciesMultiLangDatasetReader(languages=["es", "fr", "it"])
            instances = list(reader.read(str(self.data_path)))
            vocab = Vocabulary.from_instances(instances)
            iterator.index_with(vocab)
            batches = list(iterator._create_batches(instances, shuffle=False))
            assert len(batches) == 3

            for batch in batches:
                if isinstance(batch, Batch):
                    batch.index_instances(vocab)
                    batch = batch.as_tensor_dict(batch.get_padding_lengths())
                lang = ""
                for metadata in batch["metadata"]:
                    batch_lang = metadata["lang"]
                    if lang == "":
                        lang = batch_lang
                    assert lang == batch_lang
