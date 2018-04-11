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
from .constituency_parser import ConstituencyParserPredictor
from .coref import CorefPredictor
from .decomposable_attention import DecomposableAttentionPredictor
from .semantic_role_labeler import SemanticRoleLabelerPredictor
from .sentence_tagger import SentenceTaggerPredictor
from .simple_seq2seq import SimpleSeq2SeqPredictor
from .wikitables_parser import WikiTablesParserPredictor
from .nlvr_parser import NlvrParserPredictor
