from typing import Dict, List, Optional, Iterator, Tuple
import codecs
import os
import logging

from overrides import overrides
import tqdm

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import , TokenIndexer
from allennlp.data.tokenizers import Token

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

TypedSpan = Tuple[str, Tuple[int, int]] # pylint: disable=invalid-name

class OntonotesSentence:

    def __init__(self,
                 words: List[str]):
        self.words = words

class BioOntonotesSentence(OntonotesSentence):


class SpanOntonotesSentence(OntonotesSentence):






class Ontonotes(DatasetReader):
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

        # Admin for keeping track of the CONLL sentences.
        # The string id of the document. E.g "test/test/03/test_003"
        self._document_id: str = None
        # The numeric id of the sentence within this document.
        self._sentence_id: int = None

        # This first block of attributes are simple.
        # They are represented with a single item per word.

        # The words in the sentence.
        self._sentence: List[str] = []
        # The pos tags of the words in the sentence.
        self._pos_tags: List[str] = []
        # The lemmatised form of the words in the sentence which
        # have SRL or word sense information.
        self._predicate_lemmas: List[str] = []
        # The FrameNet ID of the predicate.
        self._predicate_framenet_id: List[str] = []
        # The sense of the word, if available.
        self._word_senses: List[int] = []
        # The current speaker, if available.
        self._speakers: List[str] = []

        # The fine grained NER tags for the sentence.
        self._named_entities: List[str] = []
        # the pieces of the parse tree.
        self._parse_pieces: List[str] = []

        self._verbal_predicates: List[int] = []
        self._predicate_argument_labels: List[List[str]] = []

        self._coreference = List[str] = []

    def reset(self):
        self._document_id: str = None
        self._sentence_id: int = None
        self._sentence: List[str] = []
        self._pos_tags: List[str] = []
        self._parse_pieces: List[str] = []
        self._predicate_lemmas: List[str] = []
        self._predicate_framenet_id: List[str] = []
        self._word_senses: List[int] = []
        self._speakers: List[str] = []
        self._named_entities: List[str] = []
        self._verbal_predicates: List[int] = []
        self._predicate_argument_labels: List[List[str]] = []

    def dataset_iterator(self, file_path: str) -> Iterator[str]:
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
                        self.reset()
                        continue
                    else:
                        yield self.conll_rows_to_sentence(conll_rows)
                        self.reset()

    def conll_rows_to_sentence(self, conll_rows: List[str]) -> OntonotesSentence:
        sentence: List[str] = []
        pos_tags: List[str] = []
        parse_pieces: List[str] = []
        predicate_lemmas: List[str] = []
        predicate_framenet_ids: List[str] = []
        word_senses: List[int] = []
        speakers: List[str] = []
        named_entities: List[str] = []

        coreference = List[str] = []
        verbal_predicates: List[int] = []
        predicate_argument_labels: List[List[str]] = []

        for row in conll_rows:
            conll_components = row.split()

            word = conll_components[3]
            pos_tag = conll_components[4]
            parse_piece = conll_components[5]

            parse_piece.replace("*", f" {word}")
            lemmatised_word = conll_components[6]
            framenet_id = conll_components[7]
            word_sense = conll_components[8]
            speaker = conll_components[9]
            ner_bit = conll_components[10]

            if predicate_argument_labels is None:
                predicate_argument_labels = [[] for _ in conll_components[11:-1]]

            for i, srl_component in enumerate(conll_components[10:-1]):

                component_with_word = srl_component.replace("*", f" {word}")
                predicate_argument_labels[i].append(component_with_word)

            coreference.append(conll_components[-1])

            sentence.append(word)
            pos_tags.append(pos_tag)
            parse_pieces.append(parse_piece)
            predicate_lemmas.append(lemmatised_word)
            predicate_framenet_ids.append(framenet_id)
            word_senses.append(word_sense)
            speakers.append(speaker)
            named_entities.append(ner_bit)


    def process_span_annotation(self):


    @overrides
    def read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        instances = []

        sentence: List[str] = []
        verbal_predicates: List[int] = []
        predicate_argument_labels: List[List[str]] = []
        current_span_label: List[Optional[str]] = []

        for data_file_path in self.dataset_iterator(file_path):
            with codecs.open(data_file_path, 'r', encoding='utf8') as open_file:

                conll_lines = []
                for line in open_file:
                    line = line.strip()

                    conll_lines.append(line)
                    if line == '' or line.startswith("#"):

                        # Conll format data begins and ends with lines containing a hash,
                        # which may or may not occur after an empty line. To deal with this
                        # we check if the sentence is empty or not and if it is, we just skip
                        # adding instances, because there aren't any to add.
                        if not sentence:
                            continue
                        instances.extend(self._process_sentence(sentence,
                                                                verbal_predicates,
                                                                predicate_argument_labels))
                        # Reset everything for the next sentence.
                        sentence = []
                        verbal_predicates = []
                        predicate_argument_labels = []
                        current_span_label = []
                        continue

                    conll_components = line.split()
                    word = conll_components[3]

                    sentence.append(word)
                    word_index = len(sentence) - 1
                    if word_index == 0:
                        # We're starting a new sentence. Here we set up a list of lists
                        # for the BIO labels for the annotation for each verb and create
                        # a temporary 'current_span_label' list for each annotation which
                        # we will use to keep track of whether we are beginning, inside of,
                        # or outside a particular span.
                        predicate_argument_labels = [[] for _ in conll_components[11:-1]]
                        current_span_label = [None for _ in conll_components[11:-1]]

                    num_annotations = len(predicate_argument_labels)
                    is_verbal_predicate = False
                    # Iterate over all verb annotations for the current sentence.
                    for annotation_index in range(num_annotations):
                        annotation = conll_components[11 + annotation_index]
                        label = annotation.strip("()*")

                        if "(" in annotation:
                            # Entering into a span for a particular semantic role label.
                            # We append the label and set the current span for this annotation.
                            bio_label = "B-" + label
                            predicate_argument_labels[annotation_index].append(bio_label)
                            current_span_label[annotation_index] = label

                        elif current_span_label[annotation_index] is not None:
                            # If there's no '(' token, but the current_span_label is not None,
                            # then we are inside a span.
                            bio_label = "I-" + current_span_label[annotation_index]
                            predicate_argument_labels[annotation_index].append(bio_label)
                        else:
                            # We're outside a span.
                            predicate_argument_labels[annotation_index].append("O")

                        # Exiting a span, so we reset the current span label for this annotation.
                        if ")" in annotation:
                            current_span_label[annotation_index] = None
                        # If any annotation contains this word as a verb predicate,
                        # we need to record its index. This also has the side effect
                        # of ordering the verbal predicates by their location in the
                        # sentence, automatically aligning them with the annotations.
                        if "(V" in annotation:
                            is_verbal_predicate = True

                    if is_verbal_predicate:
                        verbal_predicates.append(word_index)

        if not instances:
            raise ConfigurationError("No instances were read from the given filepath {}. "
                                     "Is the path correct?".format(file_path))
        return Dataset(instances)

    def text_to_instance(self,  # type: ignore
                         tokens: List[Token],
                         verb_label: List[int],
                         tags: List[str] = None) -> Instance:
        """
        We take `pre-tokenized` input here, along with a verb label.  The verb label should be a
        one-hot binary vector, the same length as the tokens, indicating the position of the verb
        to find arguments for.
        """
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        text_field = TextField(tokens, token_indexers=self._token_indexers)
        fields['tokens'] = text_field
        fields['verb_indicator'] = SequenceLabelField(verb_label, text_field)
        if tags:
            fields['tags'] = SequenceLabelField(tags, text_field)
        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'Ontonotes':
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        params.assert_empty(cls.__name__)
        return Ontonotes(token_indexers=token_indexers)