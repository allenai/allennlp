# allennlp.predictors.seq2seq

## Seq2SeqPredictor
```python
Seq2SeqPredictor(self, model:allennlp.models.model.Model, dataset_reader:allennlp.data.dataset_readers.dataset_reader.DatasetReader) -> None
```

Predictor for sequence to sequence models, including
:class:`~allennlp.models.encoder_decoder.composed_seq2seq` and
:class:`~allennlp.models.encoder_decoder.simple_seq2seq` and
:class:`~allennlp.models.encoder_decoder.copynet_seq2seq`.

