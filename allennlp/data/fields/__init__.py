"""
A :class:`~allennlp.data.fields.field.Field` is some piece of data instance
that ends up as an array in a model.
"""

from allennlp.data.fields.field import Field
from allennlp.data.fields.adjacency_field import AdjacencyField
from allennlp.data.fields.array_field import ArrayField
from allennlp.data.fields.flag_field import FlagField
from allennlp.data.fields.index_field import IndexField
from allennlp.data.fields.label_field import LabelField
from allennlp.data.fields.list_field import ListField
from allennlp.data.fields.metadata_field import MetadataField
from allennlp.data.fields.multilabel_field import MultiLabelField
from allennlp.data.fields.namespace_swapping_field import NamespaceSwappingField
from allennlp.data.fields.sequence_field import SequenceField
from allennlp.data.fields.sequence_label_field import SequenceLabelField
from allennlp.data.fields.span_field import SpanField
from allennlp.data.fields.text_field import TextField
