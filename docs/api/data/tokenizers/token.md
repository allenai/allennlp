# allennlp.data.tokenizers.token

## Token
```python
Token(self, /, *args, **kwargs)
```

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
type_id : ``int``, optional
    Token type id used by some pretrained language models like original BERT

    The other fields on ``Token`` follow the fields on spacy's ``Token`` object; this is one we
    added, similar to spacy's ``lex_id``.

### dep_
Alias for field number 5
### ent_type_
Alias for field number 6
### idx
Alias for field number 1
### lemma_
Alias for field number 2
### pos_
Alias for field number 3
### tag_
Alias for field number 4
### text
Alias for field number 0
### text_id
Alias for field number 7
### type_id
Alias for field number 8
