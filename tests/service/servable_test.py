# pylint: disable=no-self-use,invalid-name

from unittest import TestCase

from allennlp.service.servable import ServableCollection, Servable, JSONDict

class TestServable(Servable):
    def predict_json(self, inputs: JSONDict) -> JSONDict:
        return {"output": "test servable"}


class TestApp(TestCase):

    def setUp(self):
        self.default_models = ServableCollection.default()

    def test_list_available(self):
        available = self.default_models.list_available()
        assert available  # not empty
        assert "reverser" in available


    def test_get_model(self):
        reverser = self.default_models.get("reverser")
        assert reverser  # is not None


    def test_register(self):
        collection = ServableCollection()
        assert not collection.list_available()
        collection.register("test", TestServable())
        available = collection.list_available()
        assert available
        assert available == ["test"]
        test_servable = collection.get("test")
        assert test_servable
        result = test_servable.predict_json({})
        assert len(result) == 1
        assert result.get("output") == "test servable"

    def test_constructor(self):
        collection = ServableCollection({"test1": TestServable(), "test2": TestServable})
        available = collection.list_available()
        assert len(available) == 2
        assert "test1" in available
        assert "test2" in available
