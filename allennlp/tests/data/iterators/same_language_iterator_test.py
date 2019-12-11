from allennlp.data.dataset_readers import UniversalDependenciesMultiLangDatasetReader
from allennlp.data.iterators.same_language_iterator import SameLanguageIterator
from allennlp.common.testing import AllenNlpTestCase


class SameLanguageIteratorTest(AllenNlpTestCase):
    data_path = AllenNlpTestCase.FIXTURES_ROOT / "data" / "dependencies_multilang" / "*"

    def test_instances_of_different_languages_are_in_different_batches(self):
        reader = UniversalDependenciesMultiLangDatasetReader(languages=["es", "fr", "it"])
        iterator = SameLanguageIterator(batch_size=2, sorting_keys=[["words", "num_tokens"]])
        instances = list(reader.read(str(self.data_path)))

        batches = list(iterator._create_batches(instances, shuffle=False))
        assert len(batches) == 3

        for batch in batches:
            lang = ""
            for instance in batch:
                batch_lang = instance.fields["metadata"].metadata["lang"]
                if lang == "":
                    lang = batch_lang
                assert lang == batch_lang
