from typing import Dict, List, Optional, Iterator, Tuple
import codecs
import os
import logging

from overrides import overrides
from nltk import Tree
import tqdm


from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

TypedSpan = Tuple[str, Tuple[int, int]] # pylint: disable=invalid-name

class OntonotesSentence:

    def __init__(self,
                 document_id: str,
                 sentence_id: str,
                 words: List[str],
                 pos_tags: List[str],
                 parse_tree: Tree,
                 predicate_lemmas: List[str],
                 predicate_framenet_ids: List[str],
                 word_senses: List[int],
                 speakers: List[str],
                 named_entities: List[str],
                 srl_frames: List[List[str]]):

        self.document_id = document_id
        self.sentence_id = sentence_id
        self.words = words
        self.pos_tags = pos_tags
        self.parse_tree = parse_tree
        self.predicate_lemmas = predicate_lemmas
        self.predicate_framenet_ids = predicate_framenet_ids
        self.word_senses = word_senses
        self.speakers = speakers
        self.named_entities = named_entities
        self.srl_frames = srl_frames


class Ontonotes:
    """
    This DatasetReader is designed to read in the English OntoNotes v5.0 data
    in the format used by the CoNLL 2011/2012 shared tasks. In order to use this
    Reader, you must follow the instructions provided `here (v12 release):
    <http://cemantix.org/data/ontonotes.html>`_, which will allow you to download
    the CoNLL style annotations for the  OntoNotes v5.0 release -- LDC2013T19.tgz
    obtained from LDC.

    Once you have run the scripts on the extracted data, you will have a folder
    structured as follows:

    conll-formatted-ontonotes-5.0/
     ── data
       ├── development
           └── data
               └── english
                   └── annotations
                       ├── bc
                       ├── bn
                       ├── mz
                       ├── nw
                       ├── pt
                       ├── tc
                       └── wb
       ├── test
           └── data
               └── english
                   └── annotations
                       ├── bc
                       ├── bn
                       ├── mz
                       ├── nw
                       ├── pt
                       ├── tc
                       └── wb
       └── train
           └── data
               └── english
                   └── annotations
                       ├── bc
                       ├── bn
                       ├── mz
                       ├── nw
                       ├── pt
                       ├── tc
                       └── wb

    The file path provided to this class can then be any of the train, test or development
    directories(or the top level data directory, if you are not utilizing the splits).

    The data has the following format, ordered by column.

    1 Document ID : str
        This is a variation on the document filename
    2 Part number : int
        Some files are divided into multiple parts numbered as 000, 001, 002, ... etc.
    3 Word number : int
        This is the word index of the word in that sentence.
    4 Word : str
        This is the token as segmented/tokenized in the Treebank. Initially the ``*_skel`` file
        contain the placeholder [WORD] which gets replaced by the actual token from the
        Treebank which is part of the OntoNotes release.
    5 POS Tag : str
        This is the Penn Treebank style part of speech. When parse information is missing,
        all part of speeches except the one for which there is some sense or proposition
        annotation are marked with a XX tag. The verb is marked with just a VERB tag.
    6 Parse bit: str
        This is the bracketed structure broken before the first open parenthesis in the parse,
        and the word/part-of-speech leaf replaced with a ``*``. The full parse can be created by
        substituting the asterisk with the "([pos] [word])" string (or leaf) and concatenating
        the items in the rows of that column. When the parse information is missing, the
        first word of a sentence is tagged as ``(TOP*`` and the last word is tagged as ``*)``
        and all intermediate words are tagged with a ``*``.
    7 Predicate lemma: str
        The predicate lemma is mentioned for the rows for which we have semantic role
        information or word sense information. All other rows are marked with a "-".
    8 Predicate Frameset ID: int
        The PropBank frameset ID of the predicate in Column 7.
    9 Word sense: float
        This is the word sense of the word in Column 3.
    10 Speaker/Author: str
        This is the speaker or author name where available. Mostly in Broadcast Conversation
        and Web Log data. When not available the rows are marked with an "-".
    11 Named Entities: str
        These columns identifies the spans representing various named entities. For documents
        which do not have named entity annotation, each line is represented with an ``*``.
    12+ Predicate Arguments: str
        There is one column each of predicate argument structure information for the predicate
        mentioned in Column 7. If there are no predicates tagged in a sentence this is a
        single column with all rows marked with an ``*``.
    -1 Co-reference: str
        Co-reference chain information encoded in a parenthesis structure. For documents that do
         not have co-reference annotations, each line is represented with a "-".

    Parameters
    ----------

    Returns
    -------
    A ``Dataset`` of ``Instances`` for Semantic Role Labelling.

    """
    def __init__(self, tagging_type: str = "bio") -> None:
        self.tagging_type = tagging_type

    def datset_iterator(self, file_path) -> Iterator[OntonotesSentence]:

        for conll_file in self.dataset_path_iterator(file_path):
            yield from self.sentence_iterator(conll_file)

    def dataset_path_iterator(self, file_path: str) -> Iterator[str]:
        """
        An iterator containing file_paths in a directory
        containing CONLL-formatted files.
        """

        logger.info("Reading CONLL sentences from dataset files at: %s", file_path)
        for root, _, files in tqdm.tqdm(list(os.walk(file_path))):
            for data_file in files:
                # These are a relic of the dataset pre-processing. Every
                # file will be duplicated - one file called filename.gold_skel
                # and one generated from the preprocessing called filename.gold_conll.
                if not data_file.endswith("gold_conll"):
                    continue

                yield os.path.join(root, data_file)

    def sentence_iterator(self, file_path: str) -> Iterator[OntonotesSentence]:
        """
        An iterator over the sentences in an individual CONLL formatted file.
        """
        with codecs.open(file_path, 'r', encoding='utf8') as open_file:
            conll_rows = []
            for line in open_file:
                line = line.strip()
                if line != '' and not line.startswith('#'):
                    conll_rows.append(line)
                else:
                    if not conll_rows:
                        continue
                    else:
                        yield self.conll_rows_to_sentence(conll_rows)

    def conll_rows_to_sentence(self, conll_rows: List[str]) -> OntonotesSentence:

        # The words in the sentence.
        sentence: List[str] = []
        # The pos tags of the words in the sentence.
        pos_tags: List[str] = []
        # the pieces of the parse tree.
        parse_pieces: List[str] = []
        # The lemmatised form of the words in the sentence which
        # have SRL or word sense information.
        predicate_lemmas: List[str] = []
        # The FrameNet ID of the predicate.
        predicate_framenet_ids: List[str] = []
        # The sense of the word, if available.
        word_senses: List[int] = []
        # The current speaker, if available.
        speakers: List[str] = []

        coreference = List[str] = []
        verbal_predicates: List[str] = []
        span_labels: List[List[str]] = []
        current_span_labels: List[str] = []

        for row in conll_rows:
            conll_components = row.split()

            document_id = conll_components[0]
            sentence_id = conll_components[1]
            word = conll_components[3]
            pos_tag = conll_components[4]
            parse_piece = conll_components[5]

            # Replace brackets in text with a different
            # token for parse trees.
            if word == "(":
                parse_word = "-LRB-"
            elif word == ")":
                parse_word = "-RRB-"
            else:
                parse_word = word
            parse_piece.replace("*", f" {parse_word}")
            lemmatised_word = conll_components[6]
            framenet_id = conll_components[7]
            word_sense = conll_components[8]
            speaker = conll_components[9]

            if not span_labels:
                # Create empty lists to collect the NER and SRL BIO labels.
                span_labels = [[] for _ in conll_components[10:-1]]
                # Create variables representing the current label for each label
                # sequence we are collecting.
                current_span_labels = [None for _ in conll_components[10:-1]]

            current_span_labels, span_labels = self.process_span_annotations_for_word(conll_components[10:-1],
                                                       span_labels,
                                                       current_span_labels)

            # If any annotation contains this word as a verb predicate,
            # we need to record its index. This also has the side effect
            # of ordering the verbal predicates by their location in the
            # sentence, automatically aligning them with the annotations.
            word_is_verbal_predicate = any(["(V" in x for x in
                                           [labels[-1] for labels in span_labels[1:]]])
            if word_is_verbal_predicate:
                verbal_predicates.append(word)

            coreference.append(conll_components[-1])

            sentence.append(word)
            pos_tags.append(pos_tag)
            parse_pieces.append(parse_piece)
            predicate_lemmas.append(lemmatised_word)
            predicate_framenet_ids.append(framenet_id)
            word_senses.append(word_sense)
            speakers.append(speaker)

        named_entities = span_labels[0]
        srl_frames = span_labels[1:]
        parse_tree = Tree.fromstring(" ".join(parse_pieces))

        return OntonotesSentence(document_id,
                                 sentence_id,
                                 sentence,
                                 pos_tags,
                                 parse_tree,
                                 predicate_lemmas,
                                 predicate_framenet_ids,
                                 word_senses,
                                 speakers,
                                 named_entities,
                                 srl_frames)

    @staticmethod
    def process_span_annotations_for_word(annotations: List[str],
                                          predicate_argument_labels: List[List[str]],
                                          current_span_labels: List[Optional[str]]):

        for annotation_index in range(len(annotations)):
            annotation = annotations[annotation_index]
            label = annotation.strip("()*")

            if "(" in annotation:
                # Entering into a span for a particular semantic role label.
                # We append the label and set the current span for this annotation.
                bio_label = "B-" + label
                predicate_argument_labels[annotation_index].append(bio_label)
                current_span_labels[annotation_index] = label

            elif current_span_labels[annotation_index] is not None:
                # If there's no '(' token, but the current_span_label is not None,
                # then we are inside a span.
                bio_label = "I-" + current_span_labels[annotation_index]
                predicate_argument_labels[annotation_index].append(bio_label)
            else:
                # We're outside a span.
                predicate_argument_labels[annotation_index].append("O")

            # Exiting a span, so we reset the current span label for this annotation.
            if ")" in annotation:
                current_span_labels[annotation_index] = None

        return current_span_labels, predicate_argument_labels

    @classmethod
    def from_params(cls, params: Params) -> 'Ontonotes':
        tagging_type = params.pop("tagging_type", "bio")
        params.assert_empty(cls.__name__)
        return Ontonotes(tagging_type=tagging_type)
