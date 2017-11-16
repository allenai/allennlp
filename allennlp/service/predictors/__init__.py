"""
A :class:`~allennlp.server.predictors.predictor.Predictor` is
a wrapper for an AllenNLP ``Model``
that makes JSON predictions using JSON inputs. If you
want to serve up a model through the web service
(or using ``allennlp.commands.predict``), you'll need
a ``Predictor`` that wraps it.
"""
from .predictor import Predictor, DemoModel
from .bidaf import BidafPredictor
from .decomposable_attention import DecomposableAttentionPredictor
from .semantic_role_labeler import SemanticRoleLabelerPredictor
from .coref import CorefPredictor
from .sentence_tagger import SentenceTaggerPredictor
