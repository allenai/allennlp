# allennlp.training.callbacks.gradient_norm_and_clip

## GradientNormAndClip
```python
GradientNormAndClip(self, grad_norm:Union[float, NoneType]=None, grad_clipping:Union[float, NoneType]=None) -> None
```

Applies gradient norm and/or clipping.

Parameters
----------
grad_norm : float, optional (default = None)
    If provided, we rescale the gradients before the optimization step.
grad_clipping : float, optional (default = None)
    If provided, we use this to clip gradients in our model.

