# Models

A repository for the available models for AllenNLP.  While we highlight a particular model for
each task on https://allennlp.org/models we often have other trained models that might
work better for a particular application.


## Machine Comprehension

### [bidaf-model-2017.09.15-charpad.tar.gz](https://allennlp.s3.amazonaws.com/models/bidaf-model-2017.09.15-charpad.tar.gz) (44 MB)

Based on [BiDAF (Seo et al, 2017)](https://www.semanticscholar.org/paper/Bidirectional-Attention-Flow-for-Machine-Comprehen-Seo-Kembhavi/007ab5528b3bd310a80d553cccad4b78dc496b02)

```
$ docker run allennlp/allennlp:v0.7.0 \
    evaluate \
    https://allennlp.s3.amazonaws.com/models/bidaf-model-2017.09.15-charpad.tar.gz \
    https://allennlp.s3.amazonaws.com/datasets/squad/squad-dev-v1.1.json

Metrics:
start_acc: 0.642
  end_acc: 0.671
 span_acc: 0.552
       em: 0.683
       f1: 0.778
```

## Textual Entailment

### [decomposable-attention-elmo-2018.02.19.tar.gz](https://allennlp.s3.amazonaws.com/models/decomposable-attention-elmo-2018.02.19.tar.gz) (665 MB)

Based on [Parikh et al, 2017](https://www.semanticscholar.org/paper/A-Decomposable-Attention-Model-for-Natural-Languag-Parikh-T%C3%A4ckstr%C3%B6m/07a9478e87a8304fc3267fa16e83e9f3bbd98b27)

```
$ docker run allennlp/allennlp:v0.7.0 \
    evaluate \
    https://allennlp.s3.amazonaws.com/models/decomposable-attention-elmo-2018.02.19.tar.gz \
    https://allennlp.s3.amazonaws.com/datasets/snli/snli_1.0_test.jsonl

Metrics:
accuracy: 0.864
```

## Semantic Role Labeling

### [srl-model-2018.05.25.tar.gz](https://allennlp.s3.amazonaws.com/models/srl-model-2018.05.25.tar.gz) (697 MB)

Based on [He et al, 2017](https://www.semanticscholar.org/paper/Deep-Semantic-Role-Labeling-What-Works-and-What-s-He-Lee/a3ccff7ad63c2805078b34b8514fa9eab80d38e9)

```
f1: 0.849
```


## Coreference Resolution

### [coref-model-2018.02.05.tar.gz](https://allennlp.s3.amazonaws.com/models/coref-model-2018.02.05.tar.gz) (56 MB)

Based on [End-to-End Coreference Resolution (Lee et al, 2017)](https://www.semanticscholar.org/paper/End-to-end-Neural-Coreference-Resolution-Lee-He/3f2114893dc44eacac951f148fbff142ca200e83)

```
f1: 0.630
```

## Named Entity Recognition

### [ner-model-2018.12.18.tar.gz](https://allennlp.s3.amazonaws.com/models/ner-model-2018.12.18.tar.gz) (711.3 MB)


Based on [Deep contextualized word representations](https://arxiv.org/abs/1802.05365)

```
f1: 0.925
```

### [fine-grained-ner-model-elmo-2018.12.21.tar.gz](https://allennlp.s3.amazonaws.com/models/fine-grained-ner-model-elmo-2018.12.21.tar.gz) (724.6 MB)



## Constituency Parsing

### [elmo-constituency-parser-2018.03.14.tar.gz](https://allennlp.s3.amazonaws.com/models/elmo-constituency-parser-2018.03.14.tar.gz) (678 MB)

Based on [Minimal Span Based Constituency Parser (Stern et al, 2017)](https://www.semanticscholar.org/paper/A-Minimal-Span-Based-Neural-Constituency-Parser-Stern-Andreas/593e4e749bd2dbcaf8dc25298d830b41d435e435) but with ELMo embeddings


## Dependency Parsing

### Biaffine Parser

Based on [Dozat and Manning, 2017](https://arxiv.org/pdf/1611.01734.pdf)

* [biaffine-dependency-parser-ptb-2018.08.23.tar.gz](https://allennlp.s3.amazonaws.com/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz) (69 MB) uses [Penn Treebank](https://catalog.ldc.upenn.edu/ldc99t42) style dependencies.

* [biaffine-dependency-parser-ud-2018.08.23.tar.gz](https://allennlp.s3.amazonaws.com/models/biaffine-dependency-parser-ud-2018.08.23.tar.gz) (61 MB) uses [Universal Dependency](https://universaldependencies.org/) style depedencies.

```
f1: 0.941
```

## Semantic Parsing

### Wikitables

#### [wikitables-model-2018.09.14.tar.gz](https://allennlp.s3.amazonaws.com/models/wikitables-model-2018.09.14.tar.gz) (5 MB)

**Caveat:** that this is trained on only part of the data and not officially evaluated.

## Event2Mind

Based on [Event2Mind: Commonsense Inference on Events, Intents, and Reactions](https://homes.cs.washington.edu/~msap/debug/event2mind/docs/data/rashkin2018event2mind.pdf)
More information at: https://homes.cs.washington.edu/~msap/debug/event2mind/docs/

* [event2mind-2018.09.17.tar.gz](https://allennlp.s3.amazonaws.com/models/event2mind-2018.09.17.tar.gz) (52 MB)

```
$ allennlp evaluate \
    https://allennlp.s3.amazonaws.com/models/event2mind-2018.09.17.tar.gz  \
    https://raw.githubusercontent.com/uwnlp/event2mind/9855e83c53083b62395cc7e1af6ee9411515a14e/docs/data/test.csv

Metrics (unigram recall):
xintent: 0.36
xreact: 0.41
oreact: 0.65
```


## BiMPM

Based on [Bilateral Multi-Perspective Matching for Natural Language Sentences](https://arxiv.org/abs/1702.03814)

* [bimpm-quora-2018.08.17.tar.gz](https://allennlp.s3.amazonaws.com/models/bimpm-quora-2018.08.17.tar.gz) (147 MB)

```
```

## ESIM

Based on [Enhanced LSTM for Natural Language Inference](https://arxiv.org/pdf/1609.06038.pdf) and uses ELMo

* [esim-elmo-2018.05.17.tar.gz](https://allennlp.s3.amazonaws.com/models/esim-elmo-2018.05.17.tar.gz) (684 MB)
