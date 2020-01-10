# allennlp.modules.token_embedders.token_characters_encoder

## TokenCharactersEncoder
```python
TokenCharactersEncoder(self, embedding:allennlp.modules.token_embedders.embedding.Embedding, encoder:allennlp.modules.seq2vec_encoders.seq2vec_encoder.Seq2VecEncoder, dropout:float=0.0) -> None
```

A ``TokenCharactersEncoder`` takes the output of a
:class:`~allennlp.data.token_indexers.TokenCharactersIndexer`, which is a tensor of shape
(batch_size, num_tokens, num_characters), embeds the characters, runs a token-level encoder, and
returns the result, which is a tensor of shape (batch_size, num_tokens, encoding_dim).  We also
optionally apply dropout after the token-level encoder.

We take the embedding and encoding modules as input, so this class is itself quite simple.

