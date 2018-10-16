# pylint: disable=unused-import
import warnings
from allennlp.data.dataset_readers.semantic_parsing import WikiTablesDatasetReader

warnings.warn("allennlp.data.dataset_readers.wikitables.* has been moved."
              "Please use allennlp.data.dataset_reader.semantic_parsing.wikitables.*", FutureWarning)
