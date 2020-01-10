# allennlp.interpret.saliency_interpreters.simple_gradient

## SimpleGradient
```python
SimpleGradient(self, predictor:allennlp.predictors.predictor.Predictor) -> None
```

### saliency_interpret_from_json
```python
SimpleGradient.saliency_interpret_from_json(self, inputs:Dict[str, Any]) -> Dict[str, Any]
```

Interprets the model's prediction for inputs.  Gets the gradients of the loss with respect
to the input and returns those gradients normalized and sanitized.

