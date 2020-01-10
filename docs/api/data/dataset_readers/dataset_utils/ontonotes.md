# allennlp.data.dataset_readers.dataset_utils.ontonotes

## OntonotesSentence
```python
OntonotesSentence(self, document_id:str, sentence_id:int, words:List[str], pos_tags:List[str], parse_tree:Union[nltk.tree.Tree, NoneType], predicate_lemmas:List[Union[str, NoneType]], predicate_framenet_ids:List[Union[str, NoneType]], word_senses:List[Union[float, NoneType]], speakers:List[Union[str, NoneType]], named_entities:List[str], srl_frames:List[Tuple[str, List[str]]], coref_spans:Set[Tuple[int, Tuple[int, int]]]) -> None
```

A class representing the annotations available for a single CONLL formatted sentence.

Parameters
----------
document_id : ``str``
    This is a variation on the document filename
sentence_id : ``int``
    The integer ID of the sentence within a document.
words : ``List[str]``
    This is the tokens as segmented/tokenized in the Treebank.
pos_tags : ``List[str]``
    This is the Penn-Treebank-style part of speech. When parse information is missing,
    all parts of speech except the one for which there is some sense or proposition
    annotation are marked with a XX tag. The verb is marked with just a VERB tag.
parse_tree : ``nltk.Tree``
    An nltk Tree representing the parse. It includes POS tags as pre-terminal nodes.
    When the parse information is missing, the parse will be ``None``.
predicate_lemmas : ``List[Optional[str]]``
    The predicate lemma of the words for which we have semantic role
    information or word sense information. All other indices are ``None``.
predicate_framenet_ids : ``List[Optional[int]]``
    The PropBank frameset ID of the lemmas in ``predicate_lemmas``, or ``None``.
word_senses : ``List[Optional[float]]``
    The word senses for the words in the sentence, or ``None``. These are floats
    because the word sense can have values after the decimal, like ``1.1``.
speakers : ``List[Optional[str]]``
    The speaker information for the words in the sentence, if present, or ``None``
    This is the speaker or author name where available. Mostly in Broadcast Conversation
    and Web Log data. When not available the rows are marked with an "-".
named_entities : ``List[str]``
    The BIO tags for named entities in the sentence.
srl_frames : ``List[Tuple[str, List[str]]]``
    A dictionary keyed by the verb in the sentence for the given
    Propbank frame labels, in a BIO format.
coref_spans : ``Set[TypedSpan]``
    The spans for entity mentions involved in coreference resolution within the sentence.
    Each element is a tuple composed of (cluster_id, (start_index, end_index)). Indices
    are `inclusive`.

## Ontonotes
```python
Ontonotes(self, /, *args, **kwargs)
```

This DatasetReader is designed to read in the English OntoNotes v5.0 data
in the format used by the CoNLL 2011/2012 shared tasks. In order to use this
Reader, you must follow the instructions provided `here (v12 release):
<https://cemantix.org/data/ontonotes.html>`_, which will allow you to download
the CoNLL style annotations for the  OntoNotes v5.0 release -- LDC2013T19.tgz
obtained from LDC.

Once you have run the scripts on the extracted data, you will have a folder
structured as follows:

conll-formatted-ontonotes-5.0/
 ── data
   ├── development
       └── data
           └── english
               └── annotations
                   ├── bc
                   ├── bn
                   ├── mz
                   ├── nw
                   ├── pt
                   ├── tc
                   └── wb
   ├── test
       └── data
           └── english
               └── annotations
                   ├── bc
                   ├── bn
                   ├── mz
                   ├── nw
                   ├── pt
                   ├── tc
                   └── wb
   └── train
       └── data
           └── english
               └── annotations
                   ├── bc
                   ├── bn
                   ├── mz
                   ├── nw
                   ├── pt
                   ├── tc
                   └── wb

The file path provided to this class can then be any of the train, test or development
directories(or the top level data directory, if you are not utilizing the splits).

The data has the following format, ordered by column.

1 Document ID : ``str``
    This is a variation on the document filename
2 Part number : ``int``
    Some files are divided into multiple parts numbered as 000, 001, 002, ... etc.
3 Word number : ``int``
    This is the word index of the word in that sentence.
4 Word : ``str``
    This is the token as segmented/tokenized in the Treebank. Initially the ``*_skel`` file
    contain the placeholder [WORD] which gets replaced by the actual token from the
    Treebank which is part of the OntoNotes release.
5 POS Tag : ``str``
    This is the Penn Treebank style part of speech. When parse information is missing,
    all part of speeches except the one for which there is some sense or proposition
    annotation are marked with a XX tag. The verb is marked with just a VERB tag.
6 Parse bit : ``str``
    This is the bracketed structure broken before the first open parenthesis in the parse,
    and the word/part-of-speech leaf replaced with a ``*``. When the parse information is
    missing, the first word of a sentence is tagged as ``(TOP*`` and the last word is tagged
    as ``*)`` and all intermediate words are tagged with a ``*``.
7 Predicate lemma : ``str``
    The predicate lemma is mentioned for the rows for which we have semantic role
    information or word sense information. All other rows are marked with a "-".
8 Predicate Frameset ID : ``int``
    The PropBank frameset ID of the predicate in Column 7.
9 Word sense : ``float``
    This is the word sense of the word in Column 3.
10 Speaker/Author : ``str``
    This is the speaker or author name where available. Mostly in Broadcast Conversation
    and Web Log data. When not available the rows are marked with an "-".
11 Named Entities : ``str``
    These columns identifies the spans representing various named entities. For documents
    which do not have named entity annotation, each line is represented with an ``*``.
12+ Predicate Arguments : ``str``
    There is one column each of predicate argument structure information for the predicate
    mentioned in Column 7. If there are no predicates tagged in a sentence this is a
    single column with all rows marked with an ``*``.
-1 Co-reference : ``str``
    Co-reference chain information encoded in a parenthesis structure. For documents that do
     not have co-reference annotations, each line is represented with a "-".

### dataset_iterator
```python
Ontonotes.dataset_iterator(self, file_path:str) -> Iterator[allennlp.data.dataset_readers.dataset_utils.ontonotes.OntonotesSentence]
```

An iterator over the entire dataset, yielding all sentences processed.

### dataset_path_iterator
```python
Ontonotes.dataset_path_iterator(file_path:str) -> Iterator[str]
```

An iterator returning file_paths in a directory
containing CONLL-formatted files.

### dataset_document_iterator
```python
Ontonotes.dataset_document_iterator(self, file_path:str) -> Iterator[List[allennlp.data.dataset_readers.dataset_utils.ontonotes.OntonotesSentence]]
```

An iterator over CONLL formatted files which yields documents, regardless
of the number of document annotations in a particular file. This is useful
for conll data which has been preprocessed, such as the preprocessing which
takes place for the 2012 CONLL Coreference Resolution task.

### sentence_iterator
```python
Ontonotes.sentence_iterator(self, file_path:str) -> Iterator[allennlp.data.dataset_readers.dataset_utils.ontonotes.OntonotesSentence]
```

An iterator over the sentences in an individual CONLL formatted file.

