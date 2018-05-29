# pylint: disable=unused-import
import warnings

from allennlp.predictors.bidaf import BidafPredictor

warnings.warn("allennlp.service.predictors.* has been depreciated. "
              "Please use allennlp.predictors.*", FutureWarning)
