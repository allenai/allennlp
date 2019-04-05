from typing import NamedTuple

class Token(NamedTuple):
    """
    A simple token representation, keeping track of the token's text, offset in the passage it was
    taken from, POS tag, dependency relation, and similar information.  These fields match spacy's
    exactly, so we can just use a spacy token for this.

    Parameters
    ----------
    text : ``str``, optional
        The original text represented by this token.
    idx : ``int``, optional
        The character offset of this token into the tokenized passage.
    lemma_ : ``str``, optional
        The lemma of this token.
    pos_ : ``str``, optional
        The coarse-grained part of speech of this token.
    tag_ : ``str``, optional
        The fine-grained part of speech of this token.
    dep_ : ``str``, optional
        The dependency relation for this token.
    ent_type_ : ``str``, optional
        The entity type (i.e., the NER tag) for this token.
    text_id : ``int``, optional
        If your tokenizer returns integers instead of strings (e.g., because you're doing byte
        encoding, or some hash-based embedding), set this with the integer.  If this is set, we
        will bypass the vocabulary when indexing this token, regardless of whether ``text`` is also
        set.  You can `also` set ``text`` with the original text, if you want, so that you can
        still use a character-level representation in addition to a hash-based word embedding.

        The other fields on ``Token`` follow the fields on spacy's ``Token`` object; this is one we
        added, similar to spacy's ``lex_id``.
    """
    text: str = None
    idx: int = None
    lemma_: str = None
    pos_: str = None
    tag_: str = None
    dep_: str = None
    ent_type_: str = None
    text_id: int = None

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented


def show_token(token: Token) -> str:
    return (f"{token.text} "
            f"(idx: {token.idx}) "
            f"(lemma: {token.lemma_}) "
            f"(pos: {token.pos_}) "
            f"(tag: {token.tag_}) "
            f"(dep: {token.dep_}) "
            f"(ent_type: {token.ent_type_}) ")
