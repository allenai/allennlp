# allennlp.tools.create_elmo_embeddings_from_vocab

## main
```python
main(vocab_path:str, elmo_config_path:str, elmo_weights_path:str, output_dir:str, batch_size:int, device:int, use_custom_oov_token:bool=False)
```

Creates ELMo word representations from a vocabulary file. These
word representations are _independent_ - they are the result of running
the CNN and Highway layers of the ELMo model, but not the Bidirectional LSTM.
ELMo requires 2 additional tokens: <S> and </S>. The first token
in this file is assumed to be an unknown token.

This script produces two artifacts: A new vocabulary file
with the <S> and </S> tokens inserted and a glove formatted embedding
file containing word : vector pairs, one per line, with all values
separated by a space.

