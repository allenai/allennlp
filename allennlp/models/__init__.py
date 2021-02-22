"""
These submodules contain the classes for AllenNLP models,
all of which are subclasses of `Model`.
"""

from allennlp.models.archival import Archive, archive_model, load_archive
from allennlp.models.basic_classifier import BasicClassifier
from allennlp.models.model import Model
from allennlp.models.multitask import MultiTaskModel
from allennlp.models.simple_tagger import SimpleTagger
