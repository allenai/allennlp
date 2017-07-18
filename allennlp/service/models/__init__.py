from allennlp.service.models.pytorch import models as pytorch_models
from allennlp.service.models.placeholder import models as placeholder_models
from allennlp.service.models.simple_tagger import models as simple_tagger_models

models = {**pytorch_models(),        # pylint: disable=invalid-name
          **placeholder_models(),
          **simple_tagger_models()}
