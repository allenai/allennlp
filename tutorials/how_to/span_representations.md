
Using Span Representations in AllenNLP
--------------------------------------

Many state of the art Deep NLP models use representations of spans,
rather than representations of words, as the basic building block for
models. From AllenNLP 0.4, Span Representations are extremely easy to
use in your model.

Examples of papers which contain span representations include:

* [End to End Neural Coreference Resolution](https://arxiv.org/abs/1707.07045)
* [A Minimal Span Based Neural Constituency Parser](https://arxiv.org/abs/1705.03919)
* [Learning Recurrent Span Representations for Extractive Question Answering](https://arxiv.org/abs/1611.01436)
* [Frame-Semantic Parsing with Softmax-Margin Segmental RNNs and a Syntactic Scaffold](https://arxiv.org/abs/1706.09528)
* [Segmental Recurrent Neural Networks](https://arxiv.org/pdf/1511.06018.pdf)


## Generating `SpanFields` from text in a `DatasetReader`

`SpanFields` are a type of `Field` in AllenNLP which take a start index, an end index
and a `SequenceField` which the indices refer to. Once a batch of `SpanFields` has been
converted to a tensor, we will have a matrix of shape (batch_size, 2), where the last
dimension contains the start and end indices passed in to the SpanField constructor.


## Scoring and Pruning Spans

Span based representations have been effective for modelling/approximating structured
prediction problems - however, many models which leverage this type of representation
also involve some kind of _span enumeration_ (i.e considering all possible spans in a 
sentence/document). For a given sentence of length n, there are n^2 spans. In itself, 
this is not too problematic, but for instance, the co-reference model in AllenNLP 
compares _pairs_ of spans - meaning that naively we consider n^4 spans, with potential
document lengths of upwards of 3000 tokens.

In order to solve this problem, we need to be able to _prune_ spans as we go inside our
model. There are several ways to do this:

### Heuristically prune spans in your DatasetReader.

We have added a utility method for enumerating all spans in a sentence, but excluding
those which fulfil some condition based on the input text or any Spacy `Token` attribute.
For instance, for co-reference, all spans which are mentions (spans which are 
co-referent with _something_) never start or end with punctuation, because of the way
the Onotonotes 5.0 dataset was created. This means that we can exclude any span which
starts or ends with punctuation using a very simple python function:

```
from typing import List
from allennlp.data.dataset_readers.dataset_utils import span_utils
from allennlp.data.tokenizers.word_tokenizer import SpacyWordSplitter
from allennlp.data.tokenizers.token import Token

tokenizer = SpacyWordSplitter(pos_tags=True)
sentence = tokenizer.split_words("This is a sentence.")

def no_prefixed_punctuation(tokens: List[Token]) -> bool:
    # Only include spans which don't start or end with punctuation.
    return tokens[0].pos_ != "PUNCT" and tokens[-1].pos_ != "PUNCT"

spans = span_utils.enumerate_spans(sentence,
                                    max_span_width=3,
                                    min_span_width=2,
                                    filter_function=no_prefixed_punctuation)

# 'spans' won't include (2, 4) or (3, 4) as these have
# punctuation as their last element. Note that these spans
# have inclusive start and end indices!
assert spans == [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
```

There are other helpful functions in `allennlp.data.dataset_readers.dataset_utils.span_utils`,
such as a function to convert between BIO labelings and span based representations.

### Use a SpanPruner

It's not always possible to prune spans before they enter your model. AllenNLP contains
a `SpanPruner`, which allows you to prune spans based on a parameterised function which
is trained end-to-end with the rest of your model.

```
import torch
from torch.autograd import Variable
from allennlp.modules import SpanPruner

# Create a linear layer which will score our spans.
linear_scorer = torch.nn.Linear(5, 1)
pruner = SpanPruner(scorer=linear_scorer)

# Here we'll create some spans from a random tensor of shape
# (batch_size, num_spans, embedding_size). Typically this would
# be the output of a SpanExtractor applied to some encoded representation
# of a sentence, such as the output of an LSTM, or word embeddings.
spans = Variable(torch.randn([3, 4, 5]))
mask = Variable(torch.ones([3, 4]))

# Ok, there's quite a bit to unpack here.
# See below for a full explaination.
pruned_embeddings, pruned_mask, pruned_indices, pruned_scores = pruner(spans, mask, num_spans_to_keep=3)
```

First, we've got our `pruned_embeddings`. 
These are of shape `(batch_size, num_spans_to_keep, embedding_size)`
The spans we kept corespond to the topk with respect to the parameterised
span scorer. The other spans just get discarded, and your eventual loss
function for your model won't be a function of the discarded spans!

Secondly, we've got the `pruned_mask`, which has shape `(batch_size, num_spans_to_keep)`.
In 99% of cases, this will be all ones. However, if you have masked spans in a 
batch element, and you request that the `SpanPruner` keeps more than the number
of non-masked spans, there will be some masked elements in the returned spans.

Thirdly, we have the `pruned_indices` which has shape `(batch_size, num_spans_to_keep)`
The indices of the top-k scoring spans into the original ``spans`` tensor. 
This is returned because it can be useful to retain pointers to the original spans,
if each span is being scored by multiple distinct scorers, such as in the co-reference
model, for instance.

Finally, we have the `pruned_scores`, which has shape `(batch_size, num_spans_to_keep, 1)`.
This is returned so that you can incorporate the scores of the spans into some loss function.

## Existing AllenNLP examples for generating `SpanFields`

We've already started using `SpanFields` in AllenNLP - you can see some examples in the
[`Coreference DatasetReader`](), where we enumerate all possible spans in sentences
of a document, or in the [`PennTreeBankConstituencySpanDatasetReader`]() in order to
classify whether or not they are constituents in a constitutency parse of the sentence.

## Existing AllenNLP models which use `SpanExtractors`

Currently, both the Coreference Model and the Span Based Constituency Parser use span
representations from the output of bi-directional LSTMs. 