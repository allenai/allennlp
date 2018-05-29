# pylint: disable=unused-import
import warnings

from allennlp.predictors.semantic_role_labeler import SemanticRoleLabelerPredictor
warnings.warn("allennlp.service.predictors.* has been depreciated."
              " Please use allennlp.predictors.*", FutureWarning)
