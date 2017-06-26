from .dataset_reader import DatasetReader
from .language_modeling import LanguageModelingReader
from .snli import SnliReader
from .squad import SquadSentenceSelectionReader

dataset_readers = {}  # pylint: disable=invalid-name
dataset_readers['language modeling'] = LanguageModelingReader
dataset_readers['snli'] = SnliReader
dataset_readers['squad sentence selection'] = SquadSentenceSelectionReader
