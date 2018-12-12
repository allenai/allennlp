from typing import Dict, Any, List, Tuple

from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter

# POS tags have a unified colour.
NODE_TYPE_TO_STYLE = {}

NODE_TYPE_TO_STYLE["root"] = ["color5", "strong"]
NODE_TYPE_TO_STYLE["dep"] = ["color5", "strong"]

# Arguments
NODE_TYPE_TO_STYLE["nsubj"] = ["color1"]
NODE_TYPE_TO_STYLE["nsubjpass"] = ["color1"]
NODE_TYPE_TO_STYLE["csubj"] = ["color1"]
NODE_TYPE_TO_STYLE["csubjpass"] = ["color1"]

# Complements
NODE_TYPE_TO_STYLE["pobj"] = ["color2"]
NODE_TYPE_TO_STYLE["dobj"] = ["color2"]
NODE_TYPE_TO_STYLE["iobj"] = ["color2"]
NODE_TYPE_TO_STYLE["mark"] = ["color2"]
NODE_TYPE_TO_STYLE["pcomp"] = ["color2"]
NODE_TYPE_TO_STYLE["xcomp"] = ["color2"]
NODE_TYPE_TO_STYLE["ccomp"] = ["color2"]
NODE_TYPE_TO_STYLE["acomp"] = ["color2"]

# Modifiers
NODE_TYPE_TO_STYLE["aux"] = ["color3"]
NODE_TYPE_TO_STYLE["cop"] = ["color3"]
NODE_TYPE_TO_STYLE["det"] = ["color3"]
NODE_TYPE_TO_STYLE["conj"] = ["color3"]
NODE_TYPE_TO_STYLE["cc"] = ["color3"]
NODE_TYPE_TO_STYLE["prep"] = ["color3"]
NODE_TYPE_TO_STYLE["number"] = ["color3"]
NODE_TYPE_TO_STYLE["possesive"] = ["color3"]
NODE_TYPE_TO_STYLE["poss"] = ["color3"]
NODE_TYPE_TO_STYLE["discourse"] = ["color3"]
NODE_TYPE_TO_STYLE["expletive"] = ["color3"]
NODE_TYPE_TO_STYLE["prt"] = ["color3"]
NODE_TYPE_TO_STYLE["advcl"] = ["color3"]

NODE_TYPE_TO_STYLE["mod"] = ["color4"]
NODE_TYPE_TO_STYLE["amod"] = ["color4"]
NODE_TYPE_TO_STYLE["tmod"] = ["color4"]
NODE_TYPE_TO_STYLE["quantmod"] = ["color4"]
NODE_TYPE_TO_STYLE["npadvmod"] = ["color4"]
NODE_TYPE_TO_STYLE["infmod"] = ["color4"]
NODE_TYPE_TO_STYLE["advmod"] = ["color4"]
NODE_TYPE_TO_STYLE["appos"] = ["color4"]
NODE_TYPE_TO_STYLE["nn"] = ["color4"]

NODE_TYPE_TO_STYLE["neg"] = ["color0"]
NODE_TYPE_TO_STYLE["punct"] = ["color0"]


LINK_TO_POSITION = {}
# Put subjects on the left
LINK_TO_POSITION["nsubj"] = "left"
LINK_TO_POSITION["nsubjpass"] = "left"
LINK_TO_POSITION["csubj"] = "left"
LINK_TO_POSITION["csubjpass"] = "left"

# Put arguments and some clauses on the right
LINK_TO_POSITION["pobj"] = "right"
LINK_TO_POSITION["dobj"] = "right"
LINK_TO_POSITION["iobj"] = "right"
LINK_TO_POSITION["pcomp"] = "right"
LINK_TO_POSITION["xcomp"] = "right"
LINK_TO_POSITION["ccomp"] = "right"
LINK_TO_POSITION["acomp"] = "right"

@Predictor.register('biaffine-dependency-parser')
class BiaffineDependencyParserPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.BiaffineDependencyParser` model.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader, language: str = 'en_core_web_sm') -> None:
        super().__init__(model, dataset_reader)
        # TODO(Mark) Make the language configurable and based on a model attribute.
        self._tokenizer = SpacyWordSplitter(language=language, pos_tags=True)

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
        if self._dataset_reader.use_language_specific_pos: # type: ignore
            # fine-grained part of speech
            pos_tags = [token.tag_ for token in spacy_tokens]
        else:
            # coarse-grained part of speech (Universal Depdendencies format)
            pos_tags = [token.pos_ for token in spacy_tokens]
        return self._dataset_reader.text_to_instance(sentence_text, pos_tags)

    @overrides
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
            words = output["words"]
            pos = output["pos"]
            heads = output["predicted_heads"]
            tags = output["predicted_dependencies"]
            output["hierplane_tree"] = self._build_hierplane_tree(words, heads, tags, pos)
        return sanitize(outputs)

    @staticmethod
    def _build_hierplane_tree(words: List[str],
                              heads: List[int],
                              tags: List[str],
                              pos: List[str]) -> Dict[str, Any]:
        """
        Returns
        -------
        A JSON dictionary render-able by Hierplane for the given tree.
        """

        word_index_to_cumulative_indices: Dict[int, Tuple[int, int]] = {}
        cumulative_index = 0
        for i, word in enumerate(words):
            word_length = len(word) + 1
            word_index_to_cumulative_indices[i] = (cumulative_index, cumulative_index + word_length)
            cumulative_index += word_length

        def node_constuctor(index: int):
            children = []
            for next_index, child in enumerate(heads):
                if child == index + 1:
                    children.append(node_constuctor(next_index))

            # These are the icons which show up in the bottom right
            # corner of the node.
            attributes = [pos[index]]
            start, end = word_index_to_cumulative_indices[index]

            hierplane_node = {
                    "word": words[index],
                    # The type of the node - all nodes with the same
                    # type have a unified colour.
                    "nodeType": tags[index],
                    # Attributes of the node.
                    "attributes": attributes,
                    # The link between  the node and it's parent.
                    "link": tags[index],
                    "spans": [{"start": start, "end": end}]
            }
            if children:
                hierplane_node["children"] = children
            return hierplane_node
        # We are guaranteed that there is a single word pointing to
        # the root index, so we can find it just by searching for 0 in the list.
        root_index = heads.index(0)
        hierplane_tree = {
                "text": " ".join(words),
                "root": node_constuctor(root_index),
                "nodeTypeToStyle": NODE_TYPE_TO_STYLE,
                "linkToPosition": LINK_TO_POSITION
        }
        return hierplane_tree
