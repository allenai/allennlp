# allennlp.modules.input_variational_dropout

## InputVariationalDropout
```python
InputVariationalDropout(self, p=0.5, inplace=False)
```

Apply the dropout technique in Gal and Ghahramani, "Dropout as a Bayesian Approximation:
Representing Model Uncertainty in Deep Learning" (https://arxiv.org/abs/1506.02142) to a
3D tensor.

This module accepts a 3D tensor of shape ``(batch_size, num_timesteps, embedding_dim)``
and samples a single dropout mask of shape ``(batch_size, embedding_dim)`` and applies
it to every time step.

### forward
```python
InputVariationalDropout.forward(self, input_tensor)
```

Apply dropout to input tensor.

Parameters
----------
input_tensor : ``torch.FloatTensor``
    A tensor of shape ``(batch_size, num_timesteps, embedding_dim)``

Returns
-------
output : ``torch.FloatTensor``
    A tensor of shape ``(batch_size, num_timesteps, embedding_dim)`` with dropout applied.

