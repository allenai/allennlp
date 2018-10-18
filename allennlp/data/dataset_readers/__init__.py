"""
A :class:`~allennlp.data.dataset_readers.dataset_reader.DatasetReader`
reads a file and converts it to a collection of
:class:`~allennlp.data.instance.Instance` s.
The various subclasses know how to read specific filetypes
and produce datasets in the formats required by specific models.
"""

# pylint: disable=line-too-long
from allennlp.data.dataset_readers.ccgbank import CcgBankDatasetReader
from allennlp.data.dataset_readers.conll2003 import Conll2003DatasetReader
from allennlp.data.dataset_readers.conll2000 import Conll2000DatasetReader
from allennlp.data.dataset_readers.ontonotes_ner import OntonotesNamedEntityRecognition
from allennlp.data.dataset_readers.coreference_resolution import ConllCorefReader, WinobiasReader
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.event2mind import Event2MindDatasetReader
from allennlp.data.dataset_readers.language_modeling import LanguageModelingReader
from allennlp.data.dataset_readers.multiprocess_dataset_reader import MultiprocessDatasetReader
from allennlp.data.dataset_readers.penn_tree_bank import PennTreeBankConstituencySpanDatasetReader
from allennlp.data.dataset_readers.reading_comprehension import SquadReader, TriviaQaReader, QuACReader
from allennlp.data.dataset_readers.semantic_role_labeling import SrlReader
from allennlp.data.dataset_readers.semantic_dependency_parsing import SemanticDependenciesDatasetReader
from allennlp.data.dataset_readers.seq2seq import Seq2SeqDatasetReader
from allennlp.data.dataset_readers.sequence_tagging import SequenceTaggingDatasetReader
from allennlp.data.dataset_readers.snli import SnliReader
from allennlp.data.dataset_readers.universal_dependencies import UniversalDependenciesDatasetReader
from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import (
        StanfordSentimentTreeBankDatasetReader)
from allennlp.data.dataset_readers.quora_paraphrase import QuoraParaphraseDatasetReader
from allennlp.data.dataset_readers.semantic_parsing import (
        WikiTablesDatasetReader, AtisDatasetReader, NlvrDatasetReader, TemplateText2SqlDatasetReader)
from allennlp.data.dataset_readers.semantic_parsing.quarel import QuarelDatasetReader
from allennlp.data.dataset_readers.simple_language_modeling import SimpleLanguageModelingDatasetReader
