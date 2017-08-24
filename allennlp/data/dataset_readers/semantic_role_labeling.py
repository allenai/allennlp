import codecs
import os
import logging
from typing import Dict, List, Optional  # pylint: disable=unused-import

from overrides import overrides
import tqdm

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset import Dataset
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.common.checks import ConfigurationError

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("srl")
class SrlReader(DatasetReader):
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
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.

    Returns
    -------
    A ``Dataset`` of ``Instances`` for Semantic Role Labelling.

    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(token_indexers=token_indexers)

    def _process_sentence(self,
                          sentence: List[str],
                          verbal_predicates: List[int],
                          predicate_argument_labels: List[List[str]]) -> List[Instance]:
        """
        Parameters
        ----------
        sentence : List[str], required.
            The tokenised sentence.
        verbal_predicates : List[int], required.
            The indexes of the verbal predicates in the
            sentence which have an associated annotation.
        predicate_argument_labels : List[List[str]], required.
            A list of predicate argument labels, one for each verbal_predicate. The
            internal lists are of length: len(sentence).

        Returns
        -------
        A list of Instances.

        """
        sentence_field = TextField(sentence, self._token_indexers)
        if not verbal_predicates:
            # Sentence contains no predicates.
            tags = SequenceLabelField(["O" for _ in sentence], sentence_field)
            verb_indicator = SequenceLabelField([0 for _ in sentence], sentence_field)
            instance = Instance(fields={"tokens": sentence_field, "verb_indicator": verb_indicator, "tags": tags})
            return [instance]
        else:
            instances = []
            for verb_index, annotation in zip(verbal_predicates, predicate_argument_labels):

                tags = SequenceLabelField(annotation, sentence_field)
                indicator_ids = [0 for _ in sentence]
                indicator_ids[verb_index] = 1
                verb_indicator = SequenceLabelField(indicator_ids, sentence_field)
                instances.append(Instance(fields={"tokens": sentence_field,
                                                  "verb_indicator": verb_indicator,
                                                  "tags": tags}))
            return instances

    @overrides
    def read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        instances = []

        sentence = []  # type: List[str]
        verbal_predicates = []  # type: List[int]
        predicate_argument_labels = []  # type: List[List[str]]
        current_span_label = []  # type: List[Optional[str]]

        logger.info("Reading SRL instances from dataset files at: %s", file_path)
        for root, _, files in tqdm.tqdm(list(os.walk(file_path))):
            for data_file in files:
                # These are a relic of the dataset pre-processing. Every file will be duplicated
                # - one file called filename.gold_skel and one generated from the preprocessing
                # called filename.gold_conll.
                if 'gold_conll' not in data_file:
                    continue
                with codecs.open(os.path.join(root, data_file), 'r', encoding='utf8') as open_file:
                    for line in open_file:
                        line = line.strip()
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

    @classmethod
    def from_params(cls, params: Params) -> 'SrlReader':
        """
        Parameters
        ----------
        token_indexers: ``List[Params]``, optional
        """
        token_indexers = {}
        token_indexer_params = params.pop('token_indexers', Params({}))
        for name, indexer_params in token_indexer_params.items():
            token_indexers[name] = TokenIndexer.from_params(indexer_params)
        # The default parameters are contained within the class,
        # so if no parameters are given we must pass None.
        if token_indexers == {}:
            token_indexers = None
        params.assert_empty(cls.__name__)
        return SrlReader(token_indexers=token_indexers)
