"""
These submodules contain the classes for AllenNLP models,
all of which are subclasses of :class:`~allennlp.models.model.Model`.
"""

from allennlp.models.model import Model
from allennlp.models.archival import archive_model, load_archive, Archive
from allennlp.models.biattentive_classification_network import BiattentiveClassificationNetwork
from allennlp.models.constituency_parser import SpanConstituencyParser
from allennlp.models.biaffine_dependency_parser import BiaffineDependencyParser
from allennlp.models.coreference_resolution.coref import CoreferenceResolver
from allennlp.models.crf_tagger import CrfTagger
from allennlp.models.decomposable_attention import DecomposableAttention
from allennlp.models.event2mind import Event2Mind
from allennlp.models.encoder_decoders.simple_seq2seq import SimpleSeq2Seq
from allennlp.models.reading_comprehension.bidaf import BidirectionalAttentionFlow
from allennlp.models.reading_comprehension.naqanet import NumericallyAugmentedQaNet
from allennlp.models.reading_comprehension.qanet import QaNet
from allennlp.models.semantic_parsing.nlvr.nlvr_coverage_semantic_parser import NlvrCoverageSemanticParser
from allennlp.models.semantic_parsing.nlvr.nlvr_direct_semantic_parser import NlvrDirectSemanticParser
from allennlp.models.semantic_parsing.quarel.quarel_semantic_parser import QuarelSemanticParser
from allennlp.models.semantic_parsing.wikitables.wikitables_mml_semantic_parser import WikiTablesMmlSemanticParser
from allennlp.models.semantic_parsing.wikitables.wikitables_erm_semantic_parser import WikiTablesErmSemanticParser
from allennlp.models.semantic_parsing.atis.atis_semantic_parser import AtisSemanticParser
from allennlp.models.semantic_parsing.text2sql_parser import Text2SqlParser
from allennlp.models.semantic_role_labeler import SemanticRoleLabeler
from allennlp.models.simple_tagger import SimpleTagger
from allennlp.models.esim import ESIM
from allennlp.models.bimpm import BiMpm
from allennlp.models.graph_parser import GraphParser
from allennlp.models.bidirectional_lm import BidirectionalLanguageModel
from allennlp.models.language_model import LanguageModel
from allennlp.models.basic_classifier import BasicClassifier
