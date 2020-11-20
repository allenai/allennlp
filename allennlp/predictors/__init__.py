"""
A `Predictor` is
a wrapper for an AllenNLP `Model`
that makes JSON predictions using JSON inputs. If you
want to serve up a model through the web service
(or using `allennlp.commands.predict`), you'll need
a `Predictor` that wraps it.
"""
from allennlp.predictors.predictor import Predictor
from allennlp.predictors.sentence_tagger import SentenceTaggerPredictor
from allennlp.predictors.text_classifier import TextClassifierPredictor

try:
    from allennlp.predictors.vilbert_vqa import VilbertVqaPredictor
except ImportError:
    # VilbertVqaPredictor is not available if we don't have detectron.
    pass
