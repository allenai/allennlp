from typing import Tuple, List
from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.service.predictors.predictor import Predictor
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter


@Predictor.register('constituency-parser')
class ConstituencyParserPredictor(Predictor):
    """
    Wrapper for the :class:`~allennlp.models.SpanConstituencyParser` model.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyWordSplitter(language='en_core_web_sm')

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        """
        sentence_text = [token.text for token in self._tokenizer.split_words(json_dict["sentence"])]
        return self._dataset_reader.text_to_instance(sentence_text), {}

    @overrides
    def predict_json(self, inputs: JsonDict, cuda_device: int = -1) -> JsonDict:
        instance, return_dict = self._json_to_instance(inputs)
        outputs = self._model.forward_on_instance(instance, cuda_device)
        return_dict.update(outputs)

        # format the NLTK tree as a string on a single line.
        tree = return_dict.pop("trees")
        return_dict["trees"] = tree.pformat(margin=1000000)
        return sanitize(return_dict)

    @overrides
    def predict_batch_json(self, inputs: List[JsonDict], cuda_device: int = -1) -> List[JsonDict]:
        instances, return_dicts = zip(*self._batch_json_to_instances(inputs))
        outputs = self._model.forward_on_instances(instances, cuda_device)
        for output, return_dict in zip(outputs, return_dicts):
            return_dict.update(output)
            # format the NLTK tree as a string on a single line.
            tree = return_dict.pop("trees")
            return_dict["trees"] = tree.pformat(margin=1000000)
        return sanitize(return_dicts)
