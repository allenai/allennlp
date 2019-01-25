from typing import Dict, List, Optional, NamedTuple

import torch
from overrides import overrides

from allennlp.data.fields.field import Field
from allennlp.data.vocabulary import Vocabulary

class ProductionRule(NamedTuple):
    rule: str
    is_global_rule: bool
    rule_id: Optional[torch.LongTensor] = None
    nonterminal: Optional[str] = None

# This is just here for backward compatability.
ProductionRuleArray = ProductionRule

# mypy doesn't like that we're using a crazy data type - the data type we use here is _supposed_ to
# be in the bounds of DataArray, but ProductionRule definitely isn't.  TODO(mattg): maybe we
# should find a better way to loosen those bounds, or let people extend them.  E.g., we could have
# DataArray be a class, and let people subclass it, or something.
class ProductionRuleField(Field[ProductionRule]):  # type: ignore
    """
    This ``Field`` represents a production rule from a grammar, like "S -> [NP, VP]", "N -> John",
    or "<b,c> -> [<a,<b,c>>, a]".

    We assume a few things about how these rules are formatted:

        - There is a left-hand side (LHS) and a right-hand side (RHS), where the LHS is always a
          non-terminal, and the RHS is either a terminal, a non-terminal, or a sequence of
          terminals and/or non-terminals.
        - The LHS and the RHS are joined by " -> ", and this sequence of characters appears nowhere
          else in the rule.
        - Non-terminal sequences in the RHS are formatted as "[NT1, NT2, ...]".
        - Some rules come from a global grammar used for a whole dataset, while other rules are
          specific to a particular ``Instance``.

    We don't make use of most of these assumptions in this class, but the code that consumes this
    ``Field`` relies heavily on them in some places.

    If the given rule is in the global grammar, we treat the rule as a vocabulary item that will
    get an index and (in the model) an embedding.  If the rule is not in the global grammar, we do
    not create a vocabulary item from the rule, and don't produce a tensor for the rule - we assume
    the model will handle representing this rule in some other way.

    Because we represent global grammar rules and instance-specific rules differently, this
    ``Field`` does not lend itself well to batching its arrays, even in a sequence for a single
    training instance.  A model using this field will have to manually batch together rule
    representations after splitting apart the global rules from the ``Instance`` rules.

    In a model, this will get represented as a ``ProductionRule``, which is defined above.
    This is a namedtuple of ``(rule_string, is_global_rule, [rule_id], nonterminal)``, where the
    ``rule_id`` ``Tensor``, if present, will have shape ``(1,)``.  We don't do any batching of the
    ``Tensors``, so this gets passed to ``Model.forward()`` as a ``List[ProductionRule]``.  We
    pass along the rule string because there isn't another way to recover it for instance-specific
    rules that do not make it into the vocabulary.

    Parameters
    ----------
    rule : ``str``
        The production rule, formatted as described above.  If this field is just padding, ``rule``
        will be the empty string.
    is_global_rule : ``bool``
        Whether this rule comes from the global grammar or is an instance-specific production rule.
    vocab_namespace : ``str``, optional (default="rule_labels")
        The vocabulary namespace to use for the global production rules.  We use "rule_labels" by
        default, because we typically do not want padding and OOV tokens for these, and ending the
        namespace with "labels" means we don't get padding and OOV tokens.
    nonterminal : ``str``, optional, default = None
        The left hand side of the rule. Sometimes having this as separate part of the ``ProductionRule``
        can deduplicate work.
    """
    def __init__(self,
                 rule: str,
                 is_global_rule: bool,
                 vocab_namespace: str = 'rule_labels',
                 nonterminal: str = None) -> None:
        self.rule = rule
        self.nonterminal = nonterminal
        self.is_global_rule = is_global_rule
        self._vocab_namespace = vocab_namespace
        self._rule_id: int = None

    @overrides
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        if self.is_global_rule:
            counter[self._vocab_namespace][self.rule] += 1

    @overrides
    def index(self, vocab: Vocabulary):
        if self.is_global_rule and self._rule_id is None:
            self._rule_id = vocab.get_token_index(self.rule, self._vocab_namespace)

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        # pylint: disable=no-self-use
        return {}

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> ProductionRule:
        # pylint: disable=unused-argument
        if self.is_global_rule:
            tensor = torch.LongTensor([self._rule_id])
        else:
            tensor = None
        return ProductionRule(self.rule, self.is_global_rule, tensor, self.nonterminal)

    @overrides
    def empty_field(self): # pylint: disable=no-self-use
        # This _does_ get called, because we don't want to bother with modifying the ListField to
        # ignore padding for these.  We just make sure the rule is the empty string, which the
        # model will use to know that this rule is just padding.
        return ProductionRuleField(rule='', is_global_rule=False)

    @overrides
    def batch_tensors(self, tensor_list: List[ProductionRule]) -> List[ProductionRule]:  # type: ignore
        # pylint: disable=no-self-use
        return tensor_list

    def __str__(self) -> str:
        return f"ProductionRuleField with rule: {self.rule} (is_global_rule: " \
               f"{self.is_global_rule}) in namespace: '{self._vocab_namespace}'.'"
