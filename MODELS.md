# Models

A repository for the available models for AllenNLP.  While we highlight a particular model for
each task on https://allennlp.org/models we often have other permutations of our models that might
work better for a particular application.


## Machine Comprehension

### BiDAF

#### [bidaf-model-2017.09.15-charpad.tar.gz](https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-model-2017.09.15-charpad.tar.gz)

Based on [BiDAF (Seo et al, 2017)](https://www.semanticscholar.org/paper/Bidirectional-Attention-Flow-for-Machine-Comprehen-Seo-Kembhavi/007ab5528b3bd310a80d553cccad4b78dc496b02)

```
$ docker run allennlp/allennlp:v0.6.1 evaluate https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-model-2017.09.15-charpad.tar.gz https://s3-us-west-2.amazonaws.com/allennlp/datasets/squad/squad-dev-v1.1.json
Metrics:
start_acc: 0.6421002838221381
end_acc: 0.671050141911069
span_acc: 0.5526963103122043
em: 0.6837275307473983
f1: 0.7785736528673436
```

## Textual Entailment

### Decomposable Attention

Based on [Parikh et al, 2017](https://www.semanticscholar.org/paper/A-Decomposable-Attention-Model-for-Natural-Languag-Parikh-T%C3%A4ckstr%C3%B6m/07a9478e87a8304fc3267fa16e83e9f3bbd98b27)

#### [decomposable-attention-elmo-2018.02.19.tar.gz](https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-elmo-2018.02.19.tar.gz)

```
$ docker run allennlp/allennlp:v0.6.1 evaluate https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-elmo-2018.02.19.tar.gz https://s3-us-west-2.amazonaws.com/allennlp/datasets/snli/snli_1.0_test.jsonl
```


## Semantic Role Labeling

Based on [He et al, 2017](https://www.semanticscholar.org/paper/Deep-Semantic-Role-Labeling-What-Works-and-What-s-He-Lee/a3ccff7ad63c2805078b34b8514fa9eab80d38e9)

* [srl-model-2018.05.25.tar.gz](https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz)


## Coreference Resolution

Based on [End-to-End Coreference Resolution (Lee et al, 2017)](https://www.semanticscholar.org/paper/End-to-end-Neural-Coreference-Resolution-Lee-He/3f2114893dc44eacac951f148fbff142ca200e83)

* [coref-model-2018.02.05.tar.gz](https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz)


## Named Entity Recognition

Based on [Deep contextualized word representations](https://arxiv.org/abs/1802.05365)

* [ner-model-2018.04.26.tar.gz](https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.04.26.tar.gz)


## Constituency Parsing

Based on [Minimal Span Based Constituency Parser (Stern et al, 2017)](https://www.semanticscholar.org/paper/A-Minimal-Span-Based-Neural-Constituency-Parser-Stern-Andreas/593e4e749bd2dbcaf8dc25298d830b41d435e435) but with ELMo embeddings

* [elmo-constituency-parser-2018.03.14.tar.gz](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz)


## Dependency Parsing

### Biaffine Parser

Based on [Dozat and Manning, 2017](https://arxiv.org/pdf/1611.01734.pdf)

#### [biaffine-dependency-parser-ptb-2018.08.23.tar.gz](https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz)

[Penn Treebank](https://catalog.ldc.upenn.edu/ldc99t42) style dependencies.

#### [biaffine-dependency-parser-ud-2018.08.23.tar.gz](https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ud-2018.08.23.tar.gz)

[Universal Dependency](http://universaldependencies.org/) style depedencies.


## Semantic Parsing

### Wikitables

* [biaffine-dependency-parser-ptb-2018.08.23.tar.gz](https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz)


## Event2Mind

Based on [Event2Mind: Commonsense Inference on Events, Intents, and Reactions](https://homes.cs.washington.edu/~msap/debug/event2mind/docs/data/rashkin2018event2mind.pdf)
More information at: https://homes.cs.washington.edu/~msap/debug/event2mind/docs/

* [event2mind-2018.09.17.tar.gz](https://s3-us-west-2.amazonaws.com/allennlp/models/event2mind-2018.09.17.tar.gz)


## BiMPM

* [bimpm-quora-2018.08.17.tar.gz](https://s3-us-west-2.amazonaws.com/allennlp/datasets/quora-question-paraphrase/test.tsv)


## ESIM

Based on ???

* [esim-elmo-2018.05.17.tar.gz](https://s3-us-west-2.amazonaws.com/allennlp/models/esim-elmo-2018.05.17.tar.gz)
