"""
A :class:`~allennlp.data.dataset_readers.dataset_reader.DatasetReader`
reads a file and converts it to a
:class:`~allennlp.data.dataset.Dataset`.
The various subclasses know how to read specific filetypes
and produce datasets in the formats required by specific models.
"""

from allennlp.data.dataset_readers.conll2003 import Conll2003DatasetReader
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.language_modeling import LanguageModelingReader
from allennlp.data.dataset_readers.reading_comprehension import SquadReader, TriviaQaReader
from allennlp.data.dataset_readers.sequence_tagging import SequenceTaggingDatasetReader
from allennlp.data.dataset_readers.snli import SnliReader
from allennlp.data.dataset_readers.semantic_role_labeling import SrlReader
from allennlp.data.dataset_readers.seq2seq import Seq2SeqDatasetReader
from allennlp.data.dataset_readers.coreference_resolution import ConllCorefReader
