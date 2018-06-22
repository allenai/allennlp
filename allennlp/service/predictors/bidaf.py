# pylint: disable=unused-import
import warnings

from allennlp.predictors.bidaf import BidafPredictor

warnings.warn("allennlp.service.predictors.* has been deprecated. "
              "Please use allennlp.predictors.*", FutureWarning)
