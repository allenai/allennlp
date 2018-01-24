# pylint: disable=no-self-use,invalid-name,protected-access
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import DatasetReader, Instance, LazyDataset, Token
from allennlp.data.fields import TextField


class TestDatasetReader(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        field1 = TextField([Token(t) for t in ["this", "is", "a", "sentence", "."]], None)
        field2 = TextField([Token(t) for t in ["this", "is", "a", "different", "sentence", "."]], None)
        field3 = TextField([Token(t) for t in ["here", "is", "a", "sentence", "."]], None)
        field4 = TextField([Token(t) for t in ["this", "is", "short"]], None)
        self.instances = [Instance({"text1": field1, "text2": field2}),
                          Instance({"text1": field3, "text2": field4})]

    def get_lazy_dataset(self):
        return LazyDataset(lambda: iter(self.instances))

    def test_read_works_with_lazy_option(self):
        reader = DatasetReader(lazy=True)
        reader._read_instances = lambda x: self.get_lazy_dataset()
        dataset = reader.read('ignored')

        assert isinstance(dataset, LazyDataset)
        instances = list(iter(dataset))
        assert instances == self.instances

        # We'll make sure we can go over the dataset twice without issue.
        instances = list(iter(dataset))
        assert instances == self.instances
