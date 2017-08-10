import json
from typing import Dict, Any, Optional, List

from allennlp.common.params import Params, replace_none

import pyhocon

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
    def from_params(cls, params: Params) -> 'Servable':  # pylint: disable=unused-argument
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

    # TODO: get rid of this
    @staticmethod
    def default() -> 'ServableCollection':
        import allennlp.service.servable.models.semantic_role_labeler as semantic_role_labeler
        import allennlp.service.servable.models.bidaf as bidaf
        import allennlp.service.servable.models.decomposable_attention as decomposable_attention

        with open('experiment_config/bidaf.json') as f:
            config = json.loads(f.read())
            config['serialization_prefix'] = 'tests/fixtures/bidaf'
            config['model']['text_field_embedder']['tokens']['pretrained_file'] = \
                'tests/fixtures/glove.6B.100d.sample.txt.gz'
            bidaf_config = Params(replace_none(config))

        with open('experiment_config/semantic_role_labeler.json') as f:
            config = json.loads(f.read())
            config['serialization_prefix'] = 'tests/fixtures/srl'
            config['model']['text_field_embedder']['tokens']['pretrained_file'] = \
                'tests/fixtures/glove.6B.100d.sample.txt.gz'
            srl_config = Params(replace_none(config))

        snli_params = Params(replace_none(pyhocon.ConfigFactory.parse_file(
                'allennlp/service/servable/models/decomposable_attention.conf')))

        all_models = {
                'bidaf': bidaf.BidafServable.from_config(bidaf_config),
                'srl': semantic_role_labeler.SemanticRoleLabelerServable.from_config(srl_config),
                'snli': decomposable_attention.DecomposableAttentionServable.from_params(snli_params),
        }  # type: Dict[str, Servable]

        return ServableCollection(all_models)

    @staticmethod
    def from_params(params: Params) -> 'ServableCollection':  # pylint: disable=unused-argument
        # TODO(joelgrus) implement this
        return ServableCollection.default()
