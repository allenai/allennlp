from typing import Tuple, List

from overrides import overrides
from nltk import Tree

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
        self._tokenizer = SpacyWordSplitter(language='en_core_web_sm', pos_tags=True)

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        """
        spacy_tokens = self._tokenizer.split_words(json_dict["sentence"])
        sentence_text = [token.text for token in spacy_tokens]
        pos_tags = [token.tag_ for token in spacy_tokens]
        return self._dataset_reader.text_to_instance(sentence_text, pos_tags), {}

    @overrides
    def predict_json(self, inputs: JsonDict, cuda_device: int = -1) -> JsonDict:
        instance, return_dict = self._json_to_instance(inputs)
        outputs = self._model.forward_on_instance(instance, cuda_device)
        return_dict.update(outputs)

        # format the NLTK tree as a string on a single line.
        tree = return_dict.pop("trees")
        return_dict["hierplane_tree"] = self._build_hierplane_tree(tree, 0, is_root=True)
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
            return_dict["hierplane_tree"] = self._build_hierplane_tree(tree, 0, is_root=True)
            return_dict["trees"] = tree.pformat(margin=1000000)
        return sanitize(return_dicts)


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
                    "text": span,
                    "root": hierplane_node
            }
        return hierplane_node
