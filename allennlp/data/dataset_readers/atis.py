# pylint: disable=unused-import
import warnings
from allennlp.data.dataset_readers.semantic_parsing.atis import AtisDatasetReader

warnings.warn("allennlp.data.dataset_readers.atis.* has been moved."
              "Please use allennlp.data.dataset_reader.semantic_parsing.atis.*", FutureWarning)
