# allennlp.interpret.attackers.input_reduction

## InputReduction
```python
InputReduction(self, predictor:allennlp.predictors.predictor.Predictor, beam_size:int=3) -> None
```

Runs the input reduction method from `Pathologies of Neural Models Make Interpretations
Difficult <https://arxiv.org/abs/1804.07781>`_, which removes as many words as possible from
the input without changing the model's prediction.

The functions on this class handle a special case for NER by looking for a field called "tags"
This check is brittle, i.e., the code could break if the name of this field has changed, or if
a non-NER model has a field called "tags".

