from allennlp.common import JsonDict

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

def truncate_token(token, max_len: int = None) -> Token:
    """
    Because spacy ``Token`` s are immutable, we have to return a new Token
    """
    if max_len is None:
        return token
    else:
        return Token(text=token.text[:max_len],
                     idx=token.idx,
                     pos=token.pos_,
                     tag=token.tag_,
                     dep=token.dep_,
                     ent_type=token.ent_type_,
                     text_id=getattr(token, 'text_id', None))


def token_to_json(token: Token, short: bool = True):
    """
    Sometimes you would like to preprocess some of your data, so that
    you can do expensive operations only once across many experiments.
    Sometimes that preprocessing involves tokenizing things, in which
    case you might need to serialize / deserialize the tokens. These
    helper methods allow you to convert them to and from JSON.
    The ``short`` way is as an array [text, index]. This is good if those
    are the only fields you need. The "long" way is as a dictionary of properties.
    """
    if short:
        return [token.text, token.idx]
    else:
        blob: JsonDict = {"text": token.text}
        if token.idx is not None:
            blob['idx'] = token.idx
        if token.pos_:
            blob['pos'] = token.pos_
        if token.tag_:
            blob['tag'] = token.tag_
        if token.dep_:
            blob['dep'] = token.dep_
        if token.ent_type_:
            blob['ent_type'] = token.ent_type_
        if hasattr(token, 'text_id') and token.text_id is not None:
            blob['text_id'] = token.text_id

        return blob

def json_to_token(blob, short: bool = True, max_len: int = None) -> Token:
    if short:
        text, idx = blob
        if max_len:
            text = text[:max_len]
        return Token(text, idx)
    else:
        if max_len:
            blob['text'] = blob['text'][:max_len]
        return Token(**blob)
