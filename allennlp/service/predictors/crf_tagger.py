from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.models import Model
from allennlp.service.predictors.predictor import Predictor


@Predictor.register('crf-tagger')
class CrfTaggerPredictor(Predictor):
    """
    Wrapper for the :class:`~allennlp.models.crf_tagger.CrfTagger` model.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyWordSplitter(language='en_core_web_sm', pos_tags=True)

    @overrides
    def _json_to_instance(self, json: JsonDict) -> Instance:
        # We're overriding `predict_json` directly, so we don't need this.  But I'd rather have a
        # useless stub here then make the base class throw a RuntimeError instead of a
        # NotImplementedError - the checking on the base class is worth it.
        raise RuntimeError("this should never be called")

    @overrides
    def predict_json(self, inputs: JsonDict, cuda_device: int = -1) -> JsonDict:
        """
        Expects JSON that looks like ``{"sentence": "..."}``
        and returns JSON that looks like
        ``{"tags": [...], "words": [...]}``
        """
        sentence = inputs["sentence"]
        tokens = self._tokenizer.split_words(sentence)
        instance = self._dataset_reader.text_to_instance(tokens)
        output = self._model.forward_on_instance(instance, cuda_device)
        tags = output["tags"]

        labels = [self._model.vocab.get_token_from_index(tag, 'labels') for tag in tags]

        output = {
                "words": [token.text for token in tokens],
                "tags": labels
        }

        return sanitize(output)
