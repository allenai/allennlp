from typing import Any, Dict, List, Set, Tuple

import torch
from torch.autograd import Variable
from overrides import overrides

from allennlp.data.fields.field import Field
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.vocabulary import Vocabulary

ProductionRuleArray = Dict[str, Tuple[str, bool, Dict[str, torch.Tensor]]]  # pylint: disable=invalid-name

# mypy doesn't like that we're using a crazy data type - the data type we use here is _supposed_ to
# be in the bounds of DataArray, but ProductionRuleArray definitely isn't.  TODO(mattg): maybe we
# should find a better way to loosen those bounds, or let people extend them.  E.g., we could have
# DataArray be a class, and let people subclass it, or something.
class ProductionRuleField(Field[ProductionRuleArray]):  # type: ignore
    """
    This ``Field`` represents a production rule from a grammar, like "S -> [NP, VP]", "N -> John",
    or "<b,c> -> [<a,<b,c>>, a]".

    We assume a few things about how these rules are formatted:

        - There is a left-hand side (LHS) and a right-hand side (RHS), where the LHS is always a
          non-terminal, and the RHS is either a terminal, a non-terminal, or a sequence of
          non-terminals (for now, you can't mix terminals and non-terminals in the RHS, because we
          treat a sequence of non-terminals as if it's a `single` non-terminal for indexing
          purposes).
        - The LHS and the RHS are joined by " -> ", and this sequence of characters appears nowhere
          else in the rule.
        - Non-terminal sequences in the RHS are formatted as "[NT1, NT2, ...]" (not used here, but
          it required by some models that consume this ``Field``).
        - Terminals never begin with '<' (which indicates a functional type) or '[' (which
          indicates a non-terminal sequence).

    Given a properly-formatted production rule string, we split it into LHS and RHS, and then use
    ``TokenIndexers`` to convert these strings into data arrays.  We use (potentially) different
    ``TokenIndexers`` for terminals and non-terminals, with the assumption being that you're more
    likely to want to use character-level or featurized representations for terminals, while you
    likely will be able to just use an embedding for non-terminals.  Using ``TokenIndexers`` here
    lets us be flexible about these decisions, so you can represent and then embed the rules
    however you want to.

    While we split indexing decisions here on terminals vs. nonterminals, another reasonable way to
    split the indexing decision is on "built-in" vs. instance-specific.  That is, if you have a
    terminal production that is part of your grammar, like "<#1,#1> -> identity", you might want to
    just learn an embedding for the terminal "identity", instead of treating it like you might
    treat a terminal like "my_instance_specific_function".  You can accomplish this by just adding
    all built-in terminals to the set of ``nonterminal_types`` passed in to this ``Field`` 's
    constructor, and they will get indexed with the ``nonterminal_indexers``.

    We currently treat non-terminal sequences as a single non-terminal token from the
    ``TokenIndexers`` point of view.  The alternative here would be to represent each non-terminal
    in the sequence separately, then combine their representations in the model.  That would be
    messy, and we might allow that eventually, but for now we don't.

    Because we represent terminals and non-terminals differently, and a production rule sequence
    could have rules with both terminal and non-terminal RHSs, this ``Field`` does not lend itself
    well to batching its arrays, even in a sequence for a single training instance.  You will have
    to handle batching differently in models that use ``ProductionRuleFields``.

    In a model, this will get represented as a ``ProductionRuleArray``, which is defined above as
    ``Dict[str, Tuple[str, bool, Dict[str, torch.Tensor]]]``.  In practice, this dictionary will
    look like: ``{"left": (LHS_string, left_is_nonterminal, padded_LHS_tensor_dict),
    "right": (RHS_string, right_is_nonterminal, padded_RHS_tensor_dict)}``.

    Parameters
    ----------
    rule : ``str``
        The production rule, formatted as described above.  If this field is just padding, ``rule``
        will be the empty string.
    terminal_indexers : ``Dict[str, TokenIndexer]``
        The ``TokenIndexers`` that we will use to convert terminal strings into arrays.
    nonterminal_indexers : ``Dict[str, TokenIndexer]``
        The ``TokenIndexers`` that we will use to convert non-terminal strings into arrays.
    nonterminal_types : ``Set[str]``
        A set of basic types used in the grammar.  This is for things like NP, VP, S, N, a, b, and
        c in the examples above.  We use this set to determine whether to use the terminal or the
        non-terminal token indexers when we are representing the RHS of a production rule.
    context : Any, optional
        Additional context that can be used by the token indexers when constructing their
        representations.  This could be an utterance we're trying to produce a semantic parse for,
        for instance, and we could use this to determine whether a terminal has any overlap with
        the utterance as part of a featurized representation of the terminal.
    """
    def __init__(self,
                 rule: str,
                 terminal_indexers: Dict[str, TokenIndexer],
                 nonterminal_indexers: Dict[str, TokenIndexer],
                 nonterminal_types: Set[str],
                 context: Any = None) -> None:
        self.rule = rule
        self._terminal_indexers = terminal_indexers
        self._nonterminal_indexers = nonterminal_indexers
        self._nonterminal_types = nonterminal_types
        self._context = context

        if rule:
            self._left_side, self._right_side = rule.split(' -> ')
            self._left_side_token, self._right_side_token = (Token(self._left_side), Token(self._right_side))
            self._right_is_nonterminal = self._is_nonterminal(self._right_side)
        else:
            # This rule is just padding; we just need to make sure we return empty strings and that
            # we set `_right_is_nonterminal` to _something_.
            self._left_side = ''
            self._right_side = ''
            self._right_is_nonterminal = False
        self._right_side_indexers = nonterminal_indexers if self._right_is_nonterminal else terminal_indexers

        # mypy isn't happy with the correct annotation of Dict[str, TokenType] here, probably
        # because TokenType is a TypeVar - probably it's not valid to use TypeVars on variable
        # annotations like this?
        self._indexed_left_side: Dict = None
        self._indexed_right_side: Dict = None

    def _is_nonterminal(self, right_side: str) -> bool:
        if right_side[0] == '[' or right_side[0] == '<':
            return True
        return right_side in self._nonterminal_types

    @overrides
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        for indexer in self._nonterminal_indexers.values():
            indexer.count_vocab_items(self._left_side_token, counter)
        for indexer in self._right_side_indexers.values():
            indexer.count_vocab_items(self._right_side_token, counter)

    @overrides
    def index(self, vocab: Vocabulary):
        self._indexed_left_side = {}
        self._indexed_right_side = {}
        for indexer_name, indexer in self._nonterminal_indexers.items():
            self._indexed_left_side[indexer_name] = indexer.token_to_indices(self._left_side_token, vocab)
        for indexer_name, indexer in self._right_side_indexers.items():
            self._indexed_right_side[indexer_name] = indexer.token_to_indices(self._right_side_token, vocab)

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        padding_lengths: Dict[str, int] = {}
        if self._indexed_left_side is None:
            raise RuntimeError("You must call .index(vocab) on a field before determining padding lengths.")
        for indexer_name, indexer in self._nonterminal_indexers.items():
            indexer_lengths = indexer.get_padding_lengths(self._indexed_left_side[indexer_name])
            for padding_key, length in indexer_lengths.items():
                padding_lengths[padding_key] = max(length, padding_lengths.get(padding_key, 0))
        for indexer_name, indexer in self._right_side_indexers.items():
            indexer_lengths = indexer.get_padding_lengths(self._indexed_right_side[indexer_name])
            for padding_key, length in indexer_lengths.items():
                padding_lengths[padding_key] = max(length, padding_lengths.get(padding_key, 0))
        return padding_lengths

    @overrides
    def as_tensor(self,
                  padding_lengths: Dict[str, int],
                  cuda_device: int = -1,
                  for_training: bool = True) -> ProductionRuleArray:
        """
        Here we pad the LHS and RHS representations as necessary, but return a dictionary like:
        {"left": (self._left_side, is_nonterminal, padded_tensor),
        "right": (self._right_side, is_nonterminal, padded_tensor)}.
        This is so that you have access to the information you need to embed these representations,
        or look up valid actions given your current state.
        """
        left_side_tensors = {}
        # Because the TokenIndexers were designed to work on token sequences, and we're just giving
        # them single tokens, we need to put them into lists and then pull them out again.
        for indexer_name, indexer in self._nonterminal_indexers.items():
            padded_left_side = indexer.pad_token_sequence([self._indexed_left_side[indexer_name]],
                                                          1,
                                                          padding_lengths)[0]
            if isinstance(padded_left_side, int):
                # torch.Tensor(int) creates a tensor with shape (int,), not a tensor of shape (1,)
                # with _value_ int.  We need to force the latter.
                padded_left_side = [padded_left_side]
            tensor = Variable(torch.LongTensor(padded_left_side), volatile=not for_training)
            left_side_tensors[indexer_name] = tensor if cuda_device == -1 else tensor.cuda(cuda_device)
        right_side_tensors = {}
        for indexer_name, indexer in self._right_side_indexers.items():
            padded_right_side = indexer.pad_token_sequence([self._indexed_right_side[indexer_name]],
                                                           1,
                                                           padding_lengths)[0]
            if isinstance(padded_right_side, int):
                # torch.Tensor(int) creates a tensor with shape (int,), not a tensor of shape (1,)
                # with _value_ int.  We need to force the latter.
                padded_right_side = [padded_right_side]
            tensor = Variable(torch.LongTensor(padded_right_side), volatile=not for_training)
            right_side_tensors[indexer_name] = tensor if cuda_device == -1 else tensor.cuda(cuda_device)
        return {"left": (self._left_side, True, left_side_tensors),
                "right": (self._right_side, self._right_is_nonterminal, right_side_tensors)}

    @overrides
    def empty_field(self): # pylint: disable=no-self-use
        # Because we're not actually batching anything here, we don't need to worry about passing
        # along any of the indexers or anything, because those would only be used for padding.
        # This _does_ get called, because we don't want to bother with modifying the ListField to
        # ignore padding for these, but we don't need to do any internal padding here.  We just
        # make sure the rule is the empty string, which the model will use to know that this rule
        # is just padding.
        return ProductionRuleField(rule='',
                                   terminal_indexers={},
                                   nonterminal_indexers={},
                                   nonterminal_types=set(),
                                   context=None)

    @overrides
    def batch_tensors(self, tensor_list: List[ProductionRuleArray]) -> ProductionRuleArray:
        # pylint: disable=no-self-use
        return tensor_list  # type: ignore
