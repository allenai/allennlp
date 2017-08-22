from allennlp.common import Params
from allennlp.models import Model
from allennlp.models.archival import Archive
from allennlp.service.predictors import Predictor

def predictor_from_config(config: Params, predictor_name: str = None) -> Predictor:
    """
    Creates a ``Predictor`` using only the experiment config (and, in particular,
    no vocabulary and random weights). Some of our tests do this; you'd never want to do it
    in a non-unit-test scenario.
    """
    model = Model.load(config.duplicate())
    archive = Archive(model=model, config=config, vocab=None)
    return Predictor.from_archive(archive, predictor_name)
