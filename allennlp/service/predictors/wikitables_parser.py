# pylint: disable=unused-import
import warnings

from allennlp.predictors.wikitables_parser import WikiTablesParserPredictor
warnings.warn("allennlp.service.predictors.* has been depreciated."
              " Please use allennlp.predictors.*", FutureWarning)
