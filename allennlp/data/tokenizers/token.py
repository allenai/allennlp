class Token:
    """
    A simple token representation, keeping track of the token's text, offset in the passage it was
    taken from, POS tag, and dependency relation.  These fields match spacy's exactly, so we can
    just use a spacy token for this.

    Parameters
    ----------
    text : ``str``, optional
        The original text represented by this token.
    idx : ``int``, optional
        The character offset of this token into the tokenized passage.
    pos : ``str``, optional
        The coarse-grained part of speech of this token.
    tag : ``str``, optional
        The fine-grained part of speech of this token.
    dep : ``str``, optional
        The dependency relation for this token.
    ent_type : ``str``, optional
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
    def __init__(self,
                 text: str = None,
                 idx: int = None,
                 pos: str = None,
                 tag: str = None,
                 dep: str = None,
                 ent_type: str = None,
                 text_id: int = None) -> None:
        self.text = text
        self.idx = idx
        self.pos_ = pos
        self.tag_ = tag
        self.dep_ = dep
        self.ent_type_ = ent_type
        self.text_id = text_id

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.__str__()
