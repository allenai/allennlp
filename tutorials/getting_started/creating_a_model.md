---
layout: tutorial
title: Creating Your Own Models
id: creating-a-model
---

Using the included models is fine, but at some point you'll probably want to implement your own models,
which is what this tutorial is for.

Our [simple tagger](simple-tagger) model
uses an LSTM to capture dependencies between
the words in the input sentence, but doesn't have a great way
to capture dependencies between the _tags_. This can be a problem
for tasks like [named-entity recognition](https://en.wikipedia.org/wiki/Named-entity_recognition)
where you'd never want to (for example) have a "start of a place" tag followed by a "inside a person" tag.

We'll try to build a NER model that can outperform our simple tagger
on the [CoNLL 2003 dataset](https://www.clips.uantwerpen.be/conll2003/ner/),
which (unfortunately) you'll have to source for yourself.

The simple tagger gets about 92%
[span-based F1](https://allenai.github.io/allennlp-docs/api/allennlp.training.metrics.html#span-based-f1-measure)
on the validation dataset. We'd like to do better.

One way to approach this is to add a [Conditional Random Field](https://en.wikipedia.org/wiki/Conditional_random_field)
layer at the end of our tagging model.
(If you're not familiar with conditional random fields, [this overview paper](https://arxiv.org/abs/1011.4088)
 is helpful, as is [this PyTorch tutorial](http://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html).)

The "linear-chain" conditional random field we'll implement has a `num_tags` x `num_tags` matrix of transition costs,
where `transitions[i, j]` represents (for us) the likelihood (that is, the input to softmax) of transitioning
from the `j`-th tag to the `i`-th tag. In addition to whatever tags we're trying to predict, we have special
"start" and "end" tags that we'll stick before and after each sentence.

As this is just a component of our model, we'll implement it as a [Module](https://allenai.github.io/allennlp-docs/api/allennlp.modules.html).

## Implementing the CRF Module

To implement a PyTorch module, we just need to inherit from [`torch.nn.Module`](http://pytorch.org/docs/master/nn.html#torch.nn.Module)
and override

```python
    def forward(self, *input):
        ...
```

We'll initialize our module with the number of tags and the ids of the special start and end tags.

```python
    def __init__(self,
                 num_tags: int,
                 start_tag: int,
                 stop_tag: int) -> None:
        super().__init__()

        self.num_tags = num_tags
        self.start_tag = start_tag
        self.stop_tag = stop_tag

        # transitions[i, j] is the score for transitioning to state i from state j
        self.transitions = torch.nn.Parameter(1 * torch.randn(num_tags, num_tags))

        # We never transition to the start tag and we never transition from the stop tag
        self.transitions.data[start_tag, :] = -10000
        self.transitions.data[:, stop_tag] = -10000
```
