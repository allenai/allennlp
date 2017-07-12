from typing import Dict, cast  # pylint: disable=unused-import

from .dataset_reader import DatasetReader
from .language_modeling import LanguageModelingReader
from .snli import SnliReader
from .squad import SquadSentenceSelectionReader
from .sequence_tagging import SequenceTaggingDatasetReader

# pylint: disable=invalid-name
dataset_readers = {}  # type: Dict[str, 'DatasetReader']
# pylint: enable=invalid-name
dataset_readers['language modeling'] = cast(DatasetReader, LanguageModelingReader)
dataset_readers['snli'] = cast(DatasetReader, SnliReader)
dataset_readers['squad sentence selection'] = cast(DatasetReader, SquadSentenceSelectionReader)
dataset_readers['sequence tagging'] = cast(DatasetReader, SequenceTaggingDatasetReader)
