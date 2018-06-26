"""
Reading comprehension is loosely defined as follows: given a question and a passage of text that
contains the answer, answer the question.

These submodules contain models for things that are predominantly focused on reading comprehension.
"""

from allennlp.models.reading_comprehension.bidaf import BidirectionalAttentionFlow
from allennlp.models.reading_comprehension.bidaf_ensemble import BidafEnsemble
from allennlp.models.reading_comprehension.bidaf_self_attention import BidafPlusSelfAttention
from allennlp.models.reading_comprehension.bidaf_self_attention_bits import BidafPlusSelfAttentionBits
from allennlp.models.reading_comprehension.dqa import DQA

