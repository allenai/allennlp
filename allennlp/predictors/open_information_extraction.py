from typing import List, Dict

from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers import SpacyTokenizer
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers import Token


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
            prefix, _ = tag.split("-", 1)
            if verb_flag:
                # Continue a verb label across the different predicate parts
                prefix = "I"
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
    return [ind for ind, tag in enumerate(tags) if "V" in tag]


def get_predicate_text(sent_tokens: List[Token], tags: List[str]) -> str:
    """
    Get the predicate in this prediction.
    """
    return " ".join([sent_tokens[pred_id].text for pred_id in get_predicate_indices(tags)])


def predicates_overlap(tags1: List[str], tags2: List[str]) -> bool:
    """
    Tests whether the predicate in BIO tags1 overlap
    with those of tags2.
    """
    # Get predicate word indices from both predictions
    pred_ind1 = get_predicate_indices(tags1)
    pred_ind2 = get_predicate_indices(tags2)

    # Return if pred_ind1 pred_ind2 overlap
    return any(set.intersection(set(pred_ind1), set(pred_ind2)))


def get_coherent_next_tag(prev_label: str, cur_label: str) -> str:
    """
    Generate a coherent tag, given previous tag and current label.
    """
    if cur_label == "O":
        # Don't need to add prefix to an "O" label
        return "O"

    if prev_label == cur_label:
        return f"I-{cur_label}"
    else:
        return f"B-{cur_label}"


def merge_overlapping_predictions(tags1: List[str], tags2: List[str]) -> List[str]:
    """
    Merge two predictions into one. Assumes the predicate in tags1 overlap with
    the predicate of tags2.
    """
    ret_sequence = []
    prev_label = "O"

    # Build a coherent sequence out of two
    # spans which predicates' overlap

    for tag1, tag2 in zip(tags1, tags2):
        label1 = tag1.split("-", 1)[-1]
        label2 = tag2.split("-", 1)[-1]
        if (label1 == "V") or (label2 == "V"):
            # Construct maximal predicate length -
            # add predicate tag if any of the sequence predict it
            cur_label = "V"

        # Else - prefer an argument over 'O' label
        elif label1 != "O":
            cur_label = label1
        else:
            cur_label = label2

        # Append cur tag to the returned sequence
        cur_tag = get_coherent_next_tag(prev_label, cur_label)
        prev_label = cur_label
        ret_sequence.append(cur_tag)
    return ret_sequence


def consolidate_predictions(
    outputs: List[List[str]], sent_tokens: List[Token]
) -> Dict[str, List[str]]:
    """
    Identify that certain predicates are part of a multiword predicate
    (e.g., "decided to run") in which case, we don't need to return
    the embedded predicate ("run").
    """
    pred_dict: Dict[str, List[str]] = {}
    merged_outputs = [join_mwp(output) for output in outputs]
    predicate_texts = [get_predicate_text(sent_tokens, tags) for tags in merged_outputs]

    for pred1_text, tags1 in zip(predicate_texts, merged_outputs):
        # A flag indicating whether to add tags1 to predictions
        add_to_prediction = True

        #  Check if this predicate overlaps another predicate
        for pred2_text, tags2 in pred_dict.items():
            if predicates_overlap(tags1, tags2):
                # tags1 overlaps tags2
                pred_dict[pred2_text] = merge_overlapping_predictions(tags1, tags2)
                add_to_prediction = False

        # This predicate doesn't overlap - add as a new predicate
        if add_to_prediction:
            pred_dict[pred1_text] = tags1

    return pred_dict


def sanitize_label(label: str) -> str:
    """
    Sanitize a BIO label - this deals with OIE
    labels sometimes having some noise, as parentheses.
    """
    if "-" in label:
        prefix, suffix = label.split("-", 1)
        suffix = suffix.split("(")[-1]
        return f"{prefix}-{suffix}"
    else:
        return label


@Predictor.register("open-information-extraction")
class OpenIePredictor(Predictor):
    """
    Predictor for the [`SemanticRolelabeler`](../models/semantic_role_labeler.md) model
    (in its Open Information variant).
    Used by online demo and for prediction on an input file using command line.
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyTokenizer(pos_tags=True)

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"sentence": "...", "predicate_index": "..."}`.
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

        Expects JSON that looks like `{"sentence": "..."}`

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
        pred_ids = [i for (i, t) in enumerate(sent_tokens) if t.pos_ == "VERB"]

        # Create instances
        instances = [
            self._json_to_instance({"sentence": sent_tokens, "predicate_index": pred_id})
            for pred_id in pred_ids
        ]

        # Run model
        outputs = [
            [sanitize_label(label) for label in self._model.forward_on_instance(instance)["tags"]]
            for instance in instances
        ]

        # Consolidate predictions
        pred_dict = consolidate_predictions(outputs, sent_tokens)

        # Build and return output dictionary
        results = {"verbs": [], "words": sent_tokens}

        for tags in pred_dict.values():
            # Join multi-word predicates
            tags = join_mwp(tags)

            # Create description text
            description = make_oie_string(sent_tokens, tags)

            # Add a predicate prediction to the return dictionary.
            results["verbs"].append(
                {
                    "verb": get_predicate_text(sent_tokens, tags),
                    "description": description,
                    "tags": tags,
                }
            )

        return sanitize(results)
