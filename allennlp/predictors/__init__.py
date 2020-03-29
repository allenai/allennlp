"""
A `Predictor` is
a wrapper for an AllenNLP `Model`
that makes JSON predictions using JSON inputs. If you
want to serve up a model through the web service
(or using `allennlp.commands.predict`), you'll need
a `Predictor` that wraps it.
"""
from allennlp.predictors.predictor import Predictor
from allennlp.predictors.coref import CorefPredictor
from allennlp.predictors.decomposable_attention import DecomposableAttentionPredictor
from allennlp.predictors.sentence_tagger import SentenceTaggerPredictor
from allennlp.predictors.text_classifier import TextClassifierPredictor
