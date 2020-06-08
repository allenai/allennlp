from typing import List, Dict

from overrides import overrides
import numpy

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import FlagField, TextField, SequenceLabelField
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor


@Predictor.register("sentence_tagger")
class SentenceTaggerPredictor(Predictor):
    """
    Predictor for any model that takes in a sentence and returns
    a single set of tags for it.  In particular, it can be used with
    the [`CrfTagger`](../models/crf_tagger.md) model
    and also the [`SimpleTagger`](../models/simple_tagger.md) model.

    Registered as a `Predictor` with name "sentence_tagger".
    """

    def __init__(
        self, model: Model, dataset_reader: DatasetReader, language: str = "en_core_web_sm"
    ) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyTokenizer(language=language, pos_tags=True)

    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"sentence": "..."}`.
        Runs the underlying model, and adds the `"words"` to the output.
        """
        sentence = json_dict["sentence"]
        tokens = self._tokenizer.tokenize(sentence)
        return self._dataset_reader.text_to_instance(tokens)

    @overrides
    def predictions_to_labeled_instances(
        self, instance: Instance, outputs: Dict[str, numpy.ndarray]
    ) -> List[Instance]:
        """
        This function currently only handles BIOUL tags.

        Imagine an NER model predicts three named entities (each one with potentially
        multiple tokens). For each individual entity, we create a new Instance that has
        the label set to only that entity and the rest of the tokens are labeled as outside.
        We then return a list of those Instances.

        For example:

        ```text
        Mary  went to Seattle to visit Microsoft Research
        U-Per  O    O   U-Loc  O   O     B-Org     L-Org
        ```

        We create three instances.

        ```text
        Mary  went to Seattle to visit Microsoft Research
        U-Per  O    O    O     O   O       O         O

        Mary  went to Seattle to visit Microsoft Research
        O      O    O   U-LOC  O   O       O         O

        Mary  went to Seattle to visit Microsoft Research
        O      O    O    O     O   O     B-Org     L-Org
        ```

        We additionally add a flag to these instances to tell the model to only compute loss on
        non-O tags, so that we get gradients that are specific to the particular span prediction
        that each instance represents.
        """
        predicted_tags = outputs["tags"]
        predicted_spans = []

        i = 0
        while i < len(predicted_tags):
            tag = predicted_tags[i]
            # if its a U, add it to the list
            if tag[0] == "U":
                current_tags = [t if idx == i else "O" for idx, t in enumerate(predicted_tags)]
                predicted_spans.append(current_tags)
            # if its a B, keep going until you hit an L.
            elif tag[0] == "B":
                begin_idx = i
                while tag[0] != "L":
                    i += 1
                    tag = predicted_tags[i]
                end_idx = i
                current_tags = [
                    t if begin_idx <= idx <= end_idx else "O"
                    for idx, t in enumerate(predicted_tags)
                ]
                predicted_spans.append(current_tags)
            i += 1

        # Creates a new instance for each contiguous tag
        instances = []
        for labels in predicted_spans:
            new_instance = instance.duplicate()
            text_field: TextField = instance["tokens"]  # type: ignore
            new_instance.add_field(
                "tags", SequenceLabelField(labels, text_field), self._model.vocab
            )
            new_instance.add_field("ignore_loss_on_o_tags", FlagField(True))
            instances.append(new_instance)

        return instances
