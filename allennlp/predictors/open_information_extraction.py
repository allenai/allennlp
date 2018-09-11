from typing import List

from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers import WordTokenizer
from allennlp.models import Model
from allennlp.service.predictors.predictor import Predictor
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.tokenizers import Token

import pdb

def join_mwp(tags: List[str]) -> List[str]:
    """
    Join multi-word predicates to a single
    predicate ('V') token.
    """
    ret = []
    verb_flag = False
    for tag in tags:
        if "V" in tag:
            # Create a continuous 'V' BIO span
            prefix, _ = tag.split("-")
            if verb_flag:
                # Continue a verb label across the different predicate parts
                prefix = 'I'
            ret.append(f"{prefix}-V")
            verb_flag = True
        else:
            ret.append(tag)
            verb_flag = False

    return ret

def make_oie_string(tokens: List[Token], tags: List[str]) -> str:
    """
    Converts a list of model outputs (i.e., a list of lists of bio tags, each
    pertaining to a single word), returns an inline bracket representation of
    the prediction.
    """
    frame = []
    chunk = []
    words = [token.text for token in tokens]

    for (token, tag) in zip(words, tags):
        if tag.startswith("I-"):
            chunk.append(token)
        else:
            if chunk:
                frame.append("[" + " ".join(chunk) + "]")
                chunk = []

            if tag.startswith("B-"):
                chunk.append(tag[2:] + ": " + token)
            elif tag == "O":
                frame.append(token)

    if chunk:
        frame.append("[" + " ".join(chunk) + "]")

    return " ".join(frame)

def get_predicate_indices(tags: List[str]) -> List[int]:
    """
    Return the word indices of a predicate in BIO tags.
    """
    return [ind for (ind, tag)
            in enumerate(tags) if 'V' in tag]

def get_predicate_text(sent_tokens: List[Token], tags: List[str]) -> str:
    """
    Get the predicate in this prediction.
    """
    return " ".join([sent_tokens[pred_id].text
                     for pred_id in get_predicate_indices(tags)])

def check_predicates_subsumed(tags1: List[str], tags2: List[str]) -> bool:
    """
    Tests whether the predicate in BIO tags1 are subsumed in
    those of tags2.
    """
    # Get predicate word indices from both predictions
    pred_ind1, pred_ind2 = map(get_predicate_indices,
                               [tags1, tags2])

    # Return if pred_ind1 is contained in pred_ind2
    return (set(pred_ind1) < set(pred_ind2))

def merge_predictions(tags1: List[str], tags2: List[str]) -> List[str]:
    """
    Merge two predictions into one. Assumes the predicate in tags1 are contained in
    the predicate of tags2.
    """
    # Allow tags1 to add elements to tags2
    return [tag2 if (tag2 != 'O')\
            else tag1
            for (tag1, tag2) in zip(tags1, tags2)]

@Predictor.register('open-information-extraction')
class OpenIePredictor(Predictor):
    """
    Predictor for the :class: `models.SemanticRolelabeler` model (in its Open Information variant).
    Used by online demo and for prediction on an input file using command line.
    """
    def __init__(self,
                 model: Model,
                 dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = WordTokenizer(word_splitter=SpacyWordSplitter(pos_tags=True))


    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "...", "predicate_index": "..."}``.
        Assumes sentence is tokenized, and that predicate_index points to a specific
        predicate (word index) within the sentence, for which to produce Open IE extractions.
        """
        tokens = json_dict["sentence"]
        predicate_index = int(json_dict["predicate_index"])
        verb_labels = [0 for _ in tokens]
        verb_labels[predicate_index] = 1
        return self._dataset_reader.text_to_instance(tokens, verb_labels)

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        """
        Create instance(s) after predicting the format. One sentence containing multiple verbs
        will lead to multiple instances.

        Expects JSON that looks like ``{"sentence": "..."}``

        Returns a JSON that looks like

        .. code-block:: js

            {"tokens": [...],
             "tag_spans": [{"ARG0": "...",
                            "V": "...",
                            "ARG1": "...",
                             ...}]}
        """
        sent_tokens = self._tokenizer.tokenize(inputs["sentence"])

        # Find all verbs in the input sentence
        pred_ids = [i for (i, t)
                    in enumerate(sent_tokens)
                    if t.pos_ == "VERB"]

        # Create instances
        instances = [self._json_to_instance({"sentence": sent_tokens,
                                             "predicate_index": pred_id})
                     for pred_id in pred_ids]

        # Run model
        outputs = [self._model.forward_on_instance(instance)["tags"]
                   for instance in instances]

        # Build and return output dictionary
        results = {"verbs": [], "words": sent_tokens}

        # Merge predicates
        pred_dict = {}
        merged_outputs = list(map(join_mwp, outputs))
        for tags1 in merged_outputs:
            # A flag indicating whether to add tags1 to predictions
            add_to_prediction = True
            pred1_text = get_predicate_text(sent_tokens, tags1)
            if pred1_text in pred_dict:
                # We already added this predicate
                continue

            # Else - see if this predicate was subsumed by another predicate
            for tags2 in outputs:
                if (tags1 != tags2) and check_predicates_subsumed(tags1, tags2):
                    # tags1 is contained in tags2
                    pred2_text = get_predicate_text(sent_tokens, tags2)
                    pred_dict[pred2_text] = merge_predictions(tags1, tags2)
                    add_to_prediction = False

            if add_to_prediction:
                pred_dict[pred1_text] = tags1

        for pred_text, tags in pred_dict.items():
            # Join multi-word predicates
            tags = join_mwp(tags)

            # Create description text
            description = make_oie_string(sent_tokens, tags)

            # Add a predicate prediction to the return dictionary.
            results["verbs"].append({
                "verb": pred_text,
                "description": description,
                "tags": tags,
            })

        return sanitize(results)
