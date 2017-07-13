from typing import Dict, Type  # pylint: disable=unused-import

from allennlp.common.util import registry_decorator

# pylint: disable=invalid-name
dataset_readers = {}  # type: Dict[str, Type[DatasetReader]]
register_dataset_reader = registry_decorator("dataset reader", dataset_readers)
# pylint: enable=invalid-name

# pylint: disable=wrong-import-position
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.language_modeling import LanguageModelingReader
from allennlp.data.dataset_readers.sequence_tagging import SequenceTaggingDatasetReader
from allennlp.data.dataset_readers.snli import SnliReader
from allennlp.data.dataset_readers.squad import SquadSentenceSelectionReader
