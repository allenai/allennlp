# pylint: disable=unused-import
import warnings

from allennlp.predictors.nlvr_parser import NlvrParserPredictor
warnings.warn("allennlp.service.predictors.* has been depreciated."
              " Please use allennlp.predictors.*", FutureWarning)

