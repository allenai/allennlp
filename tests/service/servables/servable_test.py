# pylint: disable=no-self-use,invalid-name

from unittest import TestCase

from allennlp.service.servable import ServableCollection, Servable, JsonDict

class FakeServable(Servable):
    def __init__(self, *_) -> None:
        super().__init__(None, None, None, None)

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        return {"output": "test servable"}


class TestServable(TestCase):

    default_models = ServableCollection.default()

    def test_list_available(self):
        available = self.default_models.list_available()
        assert available  # not empty
        assert "bidaf" in available


    def test_get_model(self):
        bidaf = self.default_models.get("bidaf")
        assert bidaf  # is not None


    def test_register(self):
        collection = ServableCollection()
        assert not collection.list_available()
        collection.register("test", FakeServable())
        available = collection.list_available()
        assert available
        assert available == ["test"]
        test_servable = collection.get("test")
        assert test_servable
        result = test_servable.predict_json({})
        assert len(result) == 1
        assert result.get("output") == "test servable"

    def test_constructor(self):
        collection = ServableCollection({"test1": FakeServable(), "test2": FakeServable()})
        available = collection.list_available()
        assert len(available) == 2
        assert "test1" in available
        assert "test2" in available
