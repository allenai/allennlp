"""
These submodules contain the classes for AllenNLP models,
all of which are subclasses of `Model`.
"""

from allennlp.models.model import Model
from allennlp.models.archival import archive_model, load_archive, Archive
from allennlp.models.simple_tagger import SimpleTagger
from allennlp.models.basic_classifier import BasicClassifier
