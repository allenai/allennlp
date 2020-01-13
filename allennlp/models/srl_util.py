from typing import List, TextIO, Optional


def write_bio_formatted_tags_to_file(
    prediction_file: TextIO,
    gold_file: TextIO,
    verb_index: Optional[int],
    sentence: List[str],
    prediction: List[str],
    gold_labels: List[str],
):
    """
    Prints predicate argument predictions and gold labels for a single verbal
    predicate in a sentence to two provided file references.

    The CoNLL SRL format is described in
    [the shared task data README](https://www.lsi.upc.edu/~srlconll/conll05st-release/README).

    This function expects IOB2-formatted tags, where the B- tag is used in the beginning
    of every chunk (i.e. all chunks start with the B- tag).

    # Parameters

    prediction_file : TextIO, required.
        A file reference to print predictions to.
    gold_file : TextIO, required.
        A file reference to print gold labels to.
    verb_index : Optional[int], required.
        The index of the verbal predicate in the sentence which
        the gold labels are the arguments for, or None if the sentence
        contains no verbal predicate.
    sentence : List[str], required.
        The word tokens.
    prediction : List[str], required.
        The predicted BIO labels.
    gold_labels : List[str], required.
        The gold BIO labels.
    """
    conll_formatted_predictions = convert_bio_tags_to_conll_format(prediction)
    conll_formatted_gold_labels = convert_bio_tags_to_conll_format(gold_labels)
    write_conll_formatted_tags_to_file(
        prediction_file,
        gold_file,
        verb_index,
        sentence,
        conll_formatted_predictions,
        conll_formatted_gold_labels,
    )


def write_conll_formatted_tags_to_file(
    prediction_file: TextIO,
    gold_file: TextIO,
    verb_index: Optional[int],
    sentence: List[str],
    conll_formatted_predictions: List[str],
    conll_formatted_gold_labels: List[str],
):
    """
    Prints predicate argument predictions and gold labels for a single verbal
    predicate in a sentence to two provided file references.

    The CoNLL SRL format is described in
    [the shared task data README](https://www.lsi.upc.edu/~srlconll/conll05st-release/README).

    This function expects IOB2-formatted tags, where the B- tag is used in the beginning
    of every chunk (i.e. all chunks start with the B- tag).

    # Parameters

    prediction_file : TextIO, required.
        A file reference to print predictions to.
    gold_file : TextIO, required.
        A file reference to print gold labels to.
    verb_index : Optional[int], required.
        The index of the verbal predicate in the sentence which
        the gold labels are the arguments for, or None if the sentence
        contains no verbal predicate.
    sentence : List[str], required.
        The word tokens.
    conll_formatted_predictions : List[str], required.
        The predicted CoNLL-formatted labels.
    conll_formatted_gold_labels : List[str], required.
        The gold CoNLL-formatted labels.
    """
    verb_only_sentence = ["-"] * len(sentence)
    if verb_index is not None:
        verb_only_sentence[verb_index] = sentence[verb_index]

    for word, predicted, gold in zip(
        verb_only_sentence, conll_formatted_predictions, conll_formatted_gold_labels
    ):
        prediction_file.write(word.ljust(15))
        prediction_file.write(predicted.rjust(15) + "\n")
        gold_file.write(word.ljust(15))
        gold_file.write(gold.rjust(15) + "\n")
    prediction_file.write("\n")
    gold_file.write("\n")


def convert_bio_tags_to_conll_format(labels: List[str]):
    """
    Converts BIO formatted SRL tags to the format required for evaluation with the
    official CONLL 2005 perl script. Spans are represented by bracketed labels,
    with the labels of words inside spans being the same as those outside spans.
    Beginning spans always have a opening bracket and a closing asterisk (e.g. "(ARG-1*" )
    and closing spans always have a closing bracket (e.g. "*)" ). This applies even for
    length 1 spans, (e.g "(ARG-0*)").

    A full example of the conversion performed:

    [B-ARG-1, I-ARG-1, I-ARG-1, I-ARG-1, I-ARG-1, O]
    [ "(ARG-1*", "*", "*", "*", "*)", "*"]

    # Parameters

    labels : List[str], required.
        A list of BIO tags to convert to the CONLL span based format.

    # Returns

    A list of labels in the CONLL span based format.
    """
    sentence_length = len(labels)
    conll_labels = []
    for i, label in enumerate(labels):
        if label == "O":
            conll_labels.append("*")
            continue
        new_label = "*"
        # Are we at the beginning of a new span, at the first word in the sentence,
        # or is the label different from the previous one? If so, we are seeing a new label.
        if label[0] == "B" or i == 0 or label[1:] != labels[i - 1][1:]:
            new_label = "(" + label[2:] + new_label
        # Are we at the end of the sentence, is the next word a new span, or is the next
        # word not in a span? If so, we need to close the label span.
        if i == sentence_length - 1 or labels[i + 1][0] == "B" or label[1:] != labels[i + 1][1:]:
            new_label = new_label + ")"
        conll_labels.append(new_label)
    return conll_labels
