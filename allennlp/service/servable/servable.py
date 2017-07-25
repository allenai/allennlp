from typing import Dict, Any, Optional, List

JSONDict = Dict[str, Any]  # pylint: disable=invalid-name


class Servable:
    """
    Any "model" that we want to serve through either our command line tool or REST API / webapp
    needs to implement this interface.
    """
    def predict_json(self, inputs: JSONDict) -> JSONDict:
        raise NotImplementedError()

    # TODO(joelgrus): add a predict_tensor method maybe?

    @classmethod
    def from_config(cls, config: JSONDict = None) -> 'Servable':  # pylint: disable=unused-argument
        # default implementation calls no-argument constructor, you probably want to override this
        return cls()


class ServableCollection:
    """
    This represents the collection of models that are available to our command line tool or REST API.

    """
    def __init__(self, collection: Dict[str, Servable] = None) -> None:
        self.collection = collection if collection is not None else {}

    def get(self, key: str) -> Optional[Servable]:
        return self.collection.get(key)

    def register(self, key: str, servable: Servable):
        self.collection[key] = servable

    def list_available(self) -> List[str]:
        return list(self.collection.keys())

    @staticmethod
    def default() -> 'ServableCollection':
        # TODO(joelgrus): eventually get rid of this
        import allennlp.service.servable.models.pytorch as pytorch
        import allennlp.service.servable.models.placeholder as placeholder
        import allennlp.service.servable.models.simple_tagger as simple_tagger

        all_models = {
                'matrix_multiplier': pytorch.MatrixMultiplier(),
                'simple_tagger': simple_tagger.SimpleTaggerServable(),
                'reverser': placeholder.Reverser(),
                'uppercaser': placeholder.Uppercaser(),
                'lowercaser': placeholder.Lowercaser()
        }  # type: Dict[str, Servable]

        return ServableCollection(all_models)

    @staticmethod
    def from_config(config: JSONDict = None) -> 'ServableCollection':  # pylint: disable=unused-argument
        # TODO(joelgrus) implement this
        return ServableCollection.default()
