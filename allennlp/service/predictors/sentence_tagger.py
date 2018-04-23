from typing import List, Tuple
from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.models import Model
from allennlp.service.predictors.predictor import Predictor


@Predictor.register('sentence-tagger')
class SentenceTaggerPredictor(Predictor):
    """
    Predictor for any model that takes in a sentence and returns
    a single set of tags for it.  In particular, it can be used with
    the :class:`~allennlp.models.crf_tagger.CrfTagger` model
    and also
    the :class:`~allennlp.models.simple_tagger.SimpleTagger` model.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyWordSplitter(language='en_core_web_sm', pos_tags=True)

    # pylint: disable=arguments-differ
    @overrides
    def predict(self, sentence: str, cuda_device: int = -1) -> JsonDict: # type: ignore
        instance, return_dict = self._build_instance(sentence)
        outputs = self._model.forward_on_instance(instance, cuda_device)
        return_dict.update(outputs)
        return sanitize(return_dict)

    @overrides
    def predict_batch(self, inputs: List[JsonDict], cuda_device: int = -1):
        instances = [self._build_instance(**parameters) for parameters in inputs]
        return self._default_predict_batch(instances, cuda_device)

    def _build_instance(self, sentence: str) -> Tuple[Instance, JsonDict]: # type: ignore
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        Runs the underlying model, and adds the ``"words"`` to the output.
        """
        tokens = self._tokenizer.split_words(sentence)
        instance = self._dataset_reader.text_to_instance(tokens)

        return_dict: JsonDict = {"words":[token.text for token in tokens]}

        return instance, return_dict
