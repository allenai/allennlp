# pylint: disable=unused-import
import warnings
from allennlp.data.dataset_readers.semantic_parsing.nlvr import NlvrDatasetReader

warnings.warn("allennlp.data.dataset_readers.nlvr.* has been moved."
              "Please use allennlp.data.dataset_reader.semantic_parsing.nlvr.*", FutureWarning)
