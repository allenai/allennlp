from allennlp.common.util import JsonDict, sanitize
from allennlp.data.fields import TextField
from allennlp.service.predictors.predictor import Predictor


@Predictor.register('simple-tagger')
class SimpleTaggerPredictor(Predictor):
    """
    Wrapper for the :class:`~allennlp.models.bidaf.SimpleTagger` model.
    """
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        """
        Expects JSON that looks like ``{"sentence": "..."}``
        and returns JSON that looks like
        ``{"tags": [...], "class_probabilities": [[...], ..., [...]]}``
        """
        sentence = inputs["sentence"]

        tokens = TextField(self.tokenizer.tokenize(sentence)[0],
                           token_indexers=self.token_indexers)

        return sanitize(self.model.tag(tokens))
