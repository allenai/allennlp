# Using Pretrained AllenNLP Models

In this tutorial, we show how to use run the pretrained models in AllenNLP to make predictions.
This tutorial uses the Named Entity Recognition model, but the same procedure applies to any of
the models [available on our website](https://allennlp.org/models).

## Making Predictions on the Command Line

[The models page on the website](https://allennlp.org/models) lists all the models in AllenNLP,
as well as examples for how to run the model on the command line.  For example, under the
[Named Entity Recognition model](https://allennlp.org/models#named-entity-recognition) there
is a "Prediction" button that reveals the following example.

```bash
echo '{"sentence": "Did Uriah honestly think he could beat The Legend of Zelda in under three hours?"}' > ner-examples.jsonl
allennlp predict \
    https://allennlp.s3.amazonaws.com/models/ner-model-2018.04.26.tar.gz \
    ner-examples.jsonl
```

If no predictor is specified (as in the above example), then AllenNLP will use [the default
predictor](https://github.com/allenai/allennlp/blob/ea2e431cf7672fd1d04bbd382141495bfbc021f7/allennlp/service/predictors/predictor.py#L12).
You will need to create a custom predictor if you want to customize your input or output format.
Here is an example of running the above example but specifying
[the predictor](https://github.com/allenai/allennlp/blob/ea2e431cf7672fd1d04bbd382141495bfbc021f7/allennlp/service/predictors/sentence_tagger.py#L11)
explicitly.

```bash
allennlp predict \
    https://allennlp.s3.amazonaws.com/models/ner-model-2018.04.26.tar.gz \
    ner-examples.jsonl \
    --predictor sentence-tagger
```

## Making Predictions Programatically

You can also make predictions from python, using AllenNLP as a library.  The arguments will be the same as
the JSON fields in the example, and they vary by the particular predictor.  The following code example
uses the default predictor (`sentence-tagger`) for the NER model.

```python
from allennlp.predictors import Predictor
predictor = Predictor.from_path("https://allennlp.s3.amazonaws.com/models/ner-model-2018.04.26.tar.gz")
results = predictor.predict(sentence="Did Uriah honestly think he could beat The Legend of Zelda in under three hours?")
for word, tag in zip(results["words"], results["tags"]):
    print(f"{word}\t{tag}")
```

And the sample output:

```
Did	O
Uriah	U-PER
honestly	O
think	O
he	O
could	O
beat	O
The	B-MISC
Legend	I-MISC
of	I-MISC
Zelda	L-MISC
in	O
under	O
three	O
hours	O
?	O
```
