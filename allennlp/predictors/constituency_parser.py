from typing import List

from overrides import overrides
from nltk import Tree
from spacy.lang.en.tag_map import TAG_MAP

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter


# Make the links to POS tag nodes render as "pos",
# to distinguish them from constituency tags. The
# actual tag is still visible within the node.
LINK_TO_LABEL = {x: "pos" for x in TAG_MAP}

# POS tags have a unified colour.
NODE_TYPE_TO_STYLE = {x: ["color0"] for x in TAG_MAP}

# Verb and Noun phrases get their own colour.
NODE_TYPE_TO_STYLE["NP"] = ["color1"]
NODE_TYPE_TO_STYLE["NX"] = ["color1"]
NODE_TYPE_TO_STYLE["QP"] = ["color1"]
NODE_TYPE_TO_STYLE["NAC"] = ["color1"]
NODE_TYPE_TO_STYLE["VP"] = ["color2"]

# Clause level fragments
NODE_TYPE_TO_STYLE["S"] = ["color3"]
NODE_TYPE_TO_STYLE["SQ"] = ["color3"]
NODE_TYPE_TO_STYLE["SBAR"] = ["color3"]
NODE_TYPE_TO_STYLE["SBARQ"] = ["color3"]
NODE_TYPE_TO_STYLE["SINQ"] = ["color3"]
NODE_TYPE_TO_STYLE["FRAG"] = ["color3"]
NODE_TYPE_TO_STYLE["X"] = ["color3"]

# Wh-phrases.
NODE_TYPE_TO_STYLE["WHADVP"] = ["color4"]
NODE_TYPE_TO_STYLE["WHADJP"] = ["color4"]
NODE_TYPE_TO_STYLE["WHNP"] = ["color4"]
NODE_TYPE_TO_STYLE["WHPP"] = ["color4"]

# Prepositional Phrases get their own colour because
# they are linguistically interesting.
NODE_TYPE_TO_STYLE["PP"] = ["color6"]

# Everything else.
NODE_TYPE_TO_STYLE["ADJP"] = ["color5"]
NODE_TYPE_TO_STYLE["ADVP"] = ["color5"]
NODE_TYPE_TO_STYLE["CONJP"] = ["color5"]
NODE_TYPE_TO_STYLE["INTJ"] = ["color5"]
NODE_TYPE_TO_STYLE["LST"] = ["color5", "seq"]
NODE_TYPE_TO_STYLE["PRN"] = ["color5"]
NODE_TYPE_TO_STYLE["PRT"] = ["color5"]
NODE_TYPE_TO_STYLE["RRC"] = ["color5"]
NODE_TYPE_TO_STYLE["UCP"] = ["color5"]


@Predictor.register('constituency-parser')
class ConstituencyParserPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.SpanConstituencyParser` model.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader, language: str = 'en_core_web_sm') -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyWordSplitter(language=language, pos_tags=True)

    def predict(self, sentence: str) -> JsonDict:
        """
        Predict a constituency parse for the given sentence.
        Parameters
        ----------
        sentence The sentence to parse.

        Returns
        -------
        A dictionary representation of the constituency tree.
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

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)

        # format the NLTK tree as a string on a single line.
        tree = outputs.pop("trees")
        outputs["hierplane_tree"] = self._build_hierplane_tree(tree, 0, is_root=True)
        outputs["trees"] = tree.pformat(margin=1000000)
        return sanitize(outputs)

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        for output in outputs:
            # format the NLTK tree as a string on a single line.
            tree = output.pop("trees")
            output["hierplane_tree"] = self._build_hierplane_tree(tree, 0, is_root=True)
            output["trees"] = tree.pformat(margin=1000000)
        return sanitize(outputs)


    def _build_hierplane_tree(self, tree: Tree, index: int, is_root: bool) -> JsonDict:
        """
        Recursively builds a JSON dictionary from an NLTK ``Tree`` suitable for
        rendering trees using the `Hierplane library<https://allenai.github.io/hierplane/>`.

        Parameters
        ----------
        tree : ``Tree``, required.
            The tree to convert into Hierplane JSON.
        index : int, required.
            The character index into the tree, used for creating spans.
        is_root : bool
            An indicator which allows us to add the outer Hierplane JSON which
            is required for rendering.

        Returns
        -------
        A JSON dictionary render-able by Hierplane for the given tree.
        """
        children = []
        for child in tree:
            if isinstance(child, Tree):
                # If the child is a tree, it has children,
                # as NLTK leaves are just strings.
                children.append(self._build_hierplane_tree(child, index, is_root=False))
            else:
                # We're at a leaf, so add the length of
                # the word to the character index.
                index += len(child)

        label = tree.label()
        span = " ".join(tree.leaves())
        hierplane_node = {
                "word": span,
                "nodeType": label,
                "attributes": [label],
                "link": label
        }
        if children:
            hierplane_node["children"] = children
        # TODO(Mark): Figure out how to span highlighting to the leaves.
        if is_root:
            hierplane_node = {
                    "linkNameToLabel": LINK_TO_LABEL,
                    "nodeTypeToStyle": NODE_TYPE_TO_STYLE,
                    "text": span,
                    "root": hierplane_node
            }
        return hierplane_node
