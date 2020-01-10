# allennlp.data.dataset_readers.stanford_sentiment_tree_bank

## StanfordSentimentTreeBankDatasetReader
```python
StanfordSentimentTreeBankDatasetReader(self, token_indexers:Dict[str, allennlp.data.token_indexers.token_indexer.TokenIndexer]=None, use_subtrees:bool=False, granularity:str='5-class', lazy:bool=False) -> None
```

Reads tokens and their sentiment labels from the Stanford Sentiment Treebank.

The Stanford Sentiment Treebank comes with labels
from 0 to 4. ``"5-class"`` uses these labels as is. ``"3-class"`` converts the
problem into one of identifying whether a sentence is negative, positive, or
neutral sentiment. In this case, 0 and 1 are grouped as label 0 (negative sentiment),
2 is converted to label 1 (neutral sentiment) and 3 and 4 are grouped as label 2
(positive sentiment). ``"2-class"`` turns it into a binary classification problem
between positive and negative sentiment. 0 and 1 are grouped as the label 0
(negative sentiment), 2 (neutral) is discarded, and 3 and 4 are grouped as the label 1
(positive sentiment).

Expected format for each input line: a linearized tree, where nodes are labeled
by their sentiment.

The output of ``read`` is a list of ``Instance`` s with the fields:
    tokens : ``TextField`` and
    label : ``LabelField``

Parameters
----------
token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
    We use this to define the input representation for the text.  See :class:`TokenIndexer`.
use_subtrees : ``bool``, optional, (default = ``False``)
    Whether or not to use sentiment-tagged subtrees.
granularity : ``str``, optional (default = ``"5-class"``)
    One of ``"5-class"``, ``"3-class"``, or ``"2-class"``, indicating the number
    of sentiment labels to use.
lazy : ``bool``, optional, (default = ``False``)
    Whether or not instances can be read lazily.

### text_to_instance
```python
StanfordSentimentTreeBankDatasetReader.text_to_instance(self, tokens:List[str], sentiment:str=None) -> allennlp.data.instance.Instance
```

We take `pre-tokenized` input here, because we don't have a tokenizer in this class.

Parameters
----------
tokens : ``List[str]``, required.
    The tokens in a given sentence.
sentiment : ``str``, optional, (default = None).
    The sentiment for this sentence.

Returns
-------
An ``Instance`` containing the following fields:
    tokens : ``TextField``
        The tokens in the sentence or phrase.
    label : ``LabelField``
        The sentiment label of the sentence or phrase.

