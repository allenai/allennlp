# pylint: disable=unused-import
import warnings

from allennlp.predictors.coref import CorefPredictor
warnings.warn("allennlp.service.predictors.* has been depreciated. "
              " Please use allennlp.predictors.*", FutureWarning)
