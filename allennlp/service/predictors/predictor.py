# pylint: disable=unused-import
import warnings

from allennlp.predictors.predictor import Predictor, DemoModel, DEFAULT_PREDICTORS
warnings.warn("allennlp.service.predictors.* has been deprecated."
              " Please use allennlp.predictors.*", FutureWarning)
