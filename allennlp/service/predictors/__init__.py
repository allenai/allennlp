"""
A :class:`~allennlp.server.predictors.predictor.Predictor` is
a wrapper for an AllenNLP ``Model``
that makes JSON predictions using JSON inputs. If you
want to serve up a model through the web service
(or using ``allennlp.commands.predict``), you'll need
a ``Predictor`` that wraps it.
"""
from .predictor import Predictor
from .bidaf import BidafPredictor
from .decomposable_attention import DecomposableAttentionPredictor
from .semantic_role_labeler import SemanticRoleLabelerPredictor
from .simple_tagger import SimpleTaggerPredictor
from .crf_tagger import CrfTaggerPredictor
from .ontoemma import OntoEmmaPredictor