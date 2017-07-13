from typing import Dict, Type  # pylint: disable=unused-import

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.language_modeling import LanguageModelingReader
from allennlp.data.dataset_readers.snli import SnliReader
from allennlp.data.dataset_readers.squad import SquadSentenceSelectionReader
from allennlp.data.dataset_readers.sequence_tagging import SequenceTaggingDatasetReader

# pylint: disable=invalid-name
dataset_readers = {}  # type: Dict[str, Type[DatasetReader]]
# pylint: enable=invalid-name
dataset_readers['language modeling'] = LanguageModelingReader
dataset_readers['snli'] = SnliReader
dataset_readers['squad sentence selection'] = SquadSentenceSelectionReader
dataset_readers['sequence tagging'] = SequenceTaggingDatasetReader
