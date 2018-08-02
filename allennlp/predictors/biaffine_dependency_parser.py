from typing import Dict, Any, List
from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter


@Predictor.register('biaffine-dependency-parser')
class BiaffineDependencyParserPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.BiaffineDependencyParser` model.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        # TODO(Mark) Make the language configurable and based on a model attribute.
        self._tokenizer = SpacyWordSplitter(language='en_core_web_sm', pos_tags=True)

    def predict(self, sentence: str) -> JsonDict:
        """
        Predict a dependency parse for the given sentence.
        Parameters
        ----------
        sentence The sentence to parse.

        Returns
        -------
        A dictionary representation of the dependency tree.
        """
        return self.predict_json({"sentence" : sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        """
        spacy_tokens = self._tokenizer.split_words(json_dict["sentence"])
        sentence_text = [token.text for token in spacy_tokens]
        pos_tags = [token.tag_ for token in spacy_tokens]
        return self._dataset_reader.text_to_instance(sentence_text, pos_tags)


    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)

        words = outputs["words"]
        pos = outputs["pos"]
        heads = outputs["predicted_heads"]
        tags = outputs["predicted_dependencies"]
        outputs["hierplane_tree"] = self._build_hierplane_tree(words, heads, tags, pos)
        return sanitize(outputs)

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        for output in outputs:
            words = outputs["words"]
            pos = outputs["pos"]
            heads = outputs["predicted_heads"]
            tags = outputs["predicted_dependencies"]
            output["hierplane_tree"] = self._build_hierplane_tree(words, heads, tags, pos)
        return sanitize(outputs)

    @staticmethod
    def _build_hierplane_tree(words: List[str],
                              heads: List[str],
                              tags: List[str],
                              pos: List[str]) -> Dict[str, Any]:
        """
        Returns
        -------
        A JSON dictionary render-able by Hierplane for the given tree.
        """

        def node_constuctor(words, heads, tags, pos, index: int):
            children = []
            for next_index, child in enumerate(heads):
                if child == index + 1:
                    children.append(node_constuctor(words, heads, tags, pos, next_index))

            # These are the icons which show up in the bottom right
            # corner of the node. We can add anything here,
            # but for brevity we'll just add NER and a few
            # other things.
            attributes = [pos[index]]

            hierplane_node = {
                    "word": words[index],
                    # The type of the node - all nodes with the same
                    # type have a unified colour.
                    "nodeType": tags[index],
                    # Attributes of the node, eg PERSON or "email".
                    "attributes": attributes,
                    # The link between  the node and it's parent.
                    "link": tags[index],
            }
            if children:
                hierplane_node["children"] = children
            return hierplane_node

        root_index = heads.index(0)
        hierplane_tree = {
                "text": " ".join(words),
                "root": node_constuctor(words, heads, tags, pos, root_index)
        }
        return hierplane_tree
