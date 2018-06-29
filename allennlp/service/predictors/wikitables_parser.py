# pylint: disable=unused-import
import warnings

from allennlp.predictors.wikitables_parser import WikiTablesParserPredictor
warnings.warn("allennlp.service.predictors.* has been deprecated."
              " Please use allennlp.predictors.*", FutureWarning)
