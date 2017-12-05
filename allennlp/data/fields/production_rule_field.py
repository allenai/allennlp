from typing import Dict, Set, Tuple

from overrides import overrides

from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer, TokenType
from allennlp.data.vocabulary import Vocabulary

ProductionRuleArray = Dict[str, Tuple[str, bool, Dict[str, numpy.ndarray]]]

class ProductionRuleField(Field[ProductionRuleArray]):
    """
    This ``Field`` represents a production rule from a grammar, like "S -> [NP, VP]", "N -> John",
    or "<b,c> -> [<a,<b,c>>, a]".

    We assume a few things about how these rules are formatted:

        - There is a left-hand side (LHS) and a right-hand side (RHS), where the LHS is always a
          non-terminal, and the RHS is either a terminal, a non-terminal, or a sequence of
          non-terminals (for now, you can't mix terminals and non-terminals in the RHS).
        - The LHS and the RHS are joined by " -> ".
        - Non-terminal sequences in the RHS are formatted as "[NT1, NT2, ...]".
        - There are no spaces in terminals or non-terminals.
        - Terminals never begin with '<' (which indicates a functional type) or '[' (which
          indicates a non-terminal sequence).

    Given a properly-formatted production rule string, we split it into LHS and RHS, and then use
    ``TokenIndexers`` to convert these strings into data arrays.  We use (potentially) different
    ``TokenIndexers`` for terminals and non-terminals, with the assumption being that you're more
    likely to want to use character-level or featurized representations for terminals, while you
    likely will be able to just use an embedding for non-terminals.  Using ``TokenIndexers`` here
    lets us be flexible about these decisions, so you can represent and then embed the rules
    however you want to.

    We currently treat non-terminal sequences as a single non-terminal token from the
    ``TokenIndexers`` point of view.  The alternative here would be to represent each non-terminal
    in the sequence separately, then combine their representations in the model.  That would be
    messy, and we might allow that eventually, but for now we don't.

    Because we represent terminals and non-terminals differently, and a production rule sequence
    could have rules with both terminal and non-terminal RHSs, this ``Field`` does not lend itself
    well to batching its arrays, even in a sequence for a single training instance.  You will have
    to handle batching differently in models that use ``ProductionRuleFields``.

    In a model, this will get represented as a ``Dict[str, Tuple[str, bool, numpy.ndarray]]``.

    Parameters
    ----------
    rule : ``str``
        The production rule, formatted as described above.
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
        self._left_side, self._right_side = [Token(side) for side in rule.split(' -> ')]
        self._right_is_nonterminal = self._is_nonterminal(self._right_side)
        self._terminal_indexers = terminal_indexers
        self._nonterminal_indexers = nonterminal_indexers
        self._right_side_indexers = nonterminal_indexers if self._right_is_nonterminal else terminal_indexers
        self._nonterminal_types = nonterminal_types
        self._context = context
        self._indexed_left_side: Dict[str, TokenType] = None
        self._indexed_right_side: Dict[str, TokenType] = None

    def _is_nonterminal(self, right_side: str) -> bool:
        if right_side[0] == '[' or right_side[0] == '<':
            return True
        return right_side in nonterminal_types

    @overrides
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        for indexer in self._nonterminal_indexers.values():
            indexer.count_vocab_items(self._left_side, counter)
        for indexer in self._right_side_indexers.values():
            indexer.count_vocab_items(self._right_side, counter)

    @overrides
    def index(self, vocab: Vocabulary):
        self._indexed_left_side = {}
        self._indexed_right_side = {}
        for indexer_name, indexer in self._nonterminal_indexers.items():
            self._indexed_left_side[indexer_name] = indexer.token_to_indices(self._left_side, vocab)
        for indexer_name, indexer in self._right_side_indexers.items():
            self._indexed_right_side[indexer_name] = indexer.token_to_indices(self._right_side, vocab)

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        padding_lengths = {}
        if self._indexed_left_side is None:
            raise RuntimeError("You must call .index(vocab) on a field before determining padding lengths.")
        for indexer_name, indexer in self._nonterminal_indexers.items():
            for padding_key, length in indexer.get_padding_lengths(self._left_side).items()
                padding_lengths[padding_key] = max(length, padding_lengths.get(padding_key, 0))
        for indexer_name, indexer in self._right_side_indexers.items():
            for padding_key, length in indexer.get_padding_lengths(self._right_side).items()
                padding_lengths[padding_key] = max(length, padding_lengths.get(padding_key, 0))
        return padding_lengths

    @overrides
    def as_array(self, padding_lengths: Dict[str, int]) -> ProductionRuleArray:
        """
        Here we pad the LHS and RHS representations as necessary, but return a dictionary like:
        {"left": (self._left_side, is_nonterminal, padded_array),
        "right": (self._right_side, is_nonterminal, padded_array)}.
        This is so that you have access to the information you need to embed these representations,
        or look up valid actions given your current state.
        """
        left_side_arrays = {}
        # Because the TokenIndexers were designed to work on token sequences, and we're just giving
        # them single tokens, we need to put them into lists and then pull them out again.
        for indexer_name, indexer in self._nonterminal_indexers.items():
            padded_left_side = indexer.pad_token_sequence([self._indexed_left_side[indexer_name]],
                                                          1,
                                                          padding_lengths)[0]
            left_side_arrays[indexer_name] = numpy.array(padded_array)
        right_side_arrays = {}
        for indexer_name, indexer in self._right_side_indexers.items():
            padded_right_side = indexer.pad_token_sequence([self._indexed_right_side[indexer_name]],
                                                          1,
                                                          padding_lengths)[0]
            right_side_arrays[indexer_name] = numpy.array(padded_array)
        return {"left": (self._left_side, True, left_side_arrays),
                "right": (self._right_side, self._right_is_nonterminal, right_side_arrays)}

    @classmethod
    @overrides
    def batch_arrays(cls, array_list: List[ProductionRuleArray]) -> ProductionRuleArray:
        return array_list  # type: ignore
