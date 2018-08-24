============================= test session starts ==============================
platform linux -- Python 3.6.3, pytest-3.2.3, py-1.4.34, pluggy-0.4.0
rootdir: /home/eunsol/Project/quac/allennlp-quac, inifile: pytest.ini
plugins: pythonpath-0.7.1, cov-2.5.1, flaky-3.4.0
collected 3 items

allennlp/tests/models/reading_comprehension/dialog_qa_test.py F..
===Flaky Test Report===


===End Flaky Test Report===

=================================== FAILURES ===================================
______________ DialogQATest.test_batch_predictions_are_consistent ______________

self = <allennlp.tests.models.reading_comprehension.dialog_qa_test.DialogQATest testMethod=test_batch_predictions_are_consistent>

    def test_batch_predictions_are_consistent(self):
>       self.ensure_batch_predictions_are_consistent()

allennlp/tests/models/reading_comprehension/dialog_qa_test.py:37: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
allennlp/common/testing/model_test_case.py:201: in ensure_batch_predictions_are_consistent
    tensors = dataset.as_tensor_dict(dataset.get_padding_lengths())
allennlp/data/dataset.py:58: in get_padding_lengths
    for instance in self.instances]
allennlp/data/dataset.py:58: in <listcomp>
    for instance in self.instances]
allennlp/data/instance.py:69: in get_padding_lengths
    lengths[field_name] = field.get_padding_lengths()
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <allennlp.data.fields.text_field.TextField object at 0x7fef2d0e0e80>

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        """
            The ``TextField`` has a list of ``Tokens``, and each ``Token`` gets converted into arrays by
            (potentially) several ``TokenIndexers``.  This method gets the max length (over tokens)
            associated with each of these arrays.
            """
        # Our basic outline: we will iterate over `TokenIndexers`, and aggregate lengths over tokens
        # for each indexer separately.  Then we will combine the results for each indexer into a single
        # dictionary, resolving any (unlikely) key conflicts by taking a max.
        lengths = []
        if self._indexed_tokens is None:
>           raise ConfigurationError("You must call .index(vocabulary) on a "
                                     "field before determining padding lengths.")
E           allennlp.common.checks.ConfigurationError: 'You must call .index(vocabulary) on a field before determining padding lengths.'

allennlp/data/fields/text_field.py:76: ConfigurationError
----------------------------- Captured stdout call -----------------------------
<class 'allennlp.data.tokenizers.character_tokenizer.CharacterTokenizer'>
<class 'allennlp.modules.text_field_embedders.text_field_embedder.TextFieldEmbedder'>
<class 'allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder'>
<class 'allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder'>
<class 'allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder'>
<class 'allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder'>
<class 'allennlp.nn.initializers.InitializerApplicator'>
<allennlp.data.dataset_readers.dataset_reader._LazyInstances object at 0x7fef589a0518>
----------------------------- Captured stderr call -----------------------------
2018-08-24 15:53:38,288 - INFO - allennlp.common.checks - Pytorch version: 0.4.0
2018-08-24 15:53:38,302 - INFO - allennlp.common.from_params - instantiating class <class 'allennlp.data.dataset_readers.dataset_reader.DatasetReader'> from params {'lazy': True, 'num_context_answers': 2, 'token_indexers': {'token_characters': {'character_tokenizer': {'byte_encoding': 'utf-8', 'end_tokens': [260], 'start_tokens': [259]}, 'type': 'characters'}}, 'type': 'quac'} and extras {}
2018-08-24 15:53:38,302 - INFO - allennlp.common.from_params - instantiating class <class 'allennlp.data.dataset_readers.reading_comprehension.quac.QuACReader'> from params {'lazy': True, 'num_context_answers': 2, 'token_indexers': {'token_characters': {'character_tokenizer': {'byte_encoding': 'utf-8', 'end_tokens': [260], 'start_tokens': [259]}, 'type': 'characters'}}} and extras {}
2018-08-24 15:53:38,302 - INFO - allennlp.common.from_params - instantiating class allennlp.data.token_indexers.token_indexer.TokenIndexer from params {'character_tokenizer': {'byte_encoding': 'utf-8', 'end_tokens': [260], 'start_tokens': [259]}, 'type': 'characters'} and extras {}
2018-08-24 15:53:38,302 - INFO - allennlp.common.from_params - instantiating class allennlp.data.token_indexers.token_characters_indexer.TokenCharactersIndexer from params {'character_tokenizer': {'byte_encoding': 'utf-8', 'end_tokens': [260], 'start_tokens': [259]}} and extras {}
2018-08-24 15:53:38,302 - INFO - allennlp.common.from_params - instantiating class <class 'allennlp.data.tokenizers.character_tokenizer.CharacterTokenizer'> from params {'byte_encoding': 'utf-8', 'end_tokens': [260], 'start_tokens': [259]} and extras {}
2018-08-24 15:53:38,531 - INFO - allennlp.data.vocabulary - Fitting token dictionary from dataset.
0it [00:00, ?it/s]2018-08-24 15:53:38,531 - INFO - allennlp.data.dataset_readers.reading_comprehension.quac - Reading file at /home/eunsol/Project/quac/allennlp-quac/allennlp/tests/fixtures/data/quac_sample.json
2018-08-24 15:53:38,531 - INFO - allennlp.data.dataset_readers.reading_comprehension.quac - Reading the dataset
2it [00:00, 78.30it/s]
2018-08-24 15:53:38,557 - INFO - allennlp.common.from_params - instantiating class <class 'allennlp.models.model.Model'> from params {'dropout': 0.3, 'initializer': [], 'marker_embedding_dim': 10, 'num_context_answers': 2, 'phrase_layer': {'bidirectional': True, 'hidden_size': 10, 'input_size': 120, 'num_layers': 1, 'type': 'gru'}, 'residual_encoder': {'bidirectional': True, 'hidden_size': 10, 'input_size': 20, 'num_layers': 1, 'type': 'gru'}, 'span_end_encoder': {'bidirectional': True, 'hidden_size': 10, 'input_size': 40, 'num_layers': 1, 'type': 'gru'}, 'span_start_encoder': {'bidirectional': True, 'hidden_size': 10, 'input_size': 20, 'num_layers': 1, 'type': 'gru'}, 'text_field_embedder': {'token_characters': {'dropout': 0.2, 'embedding': {'embedding_dim': 20, 'num_embeddings': 262}, 'encoder': {'embedding_dim': 20, 'ngram_filter_sizes': [5], 'num_filters': 100, 'type': 'cnn'}, 'type': 'character_encoding'}}, 'type': 'dialog_qa'} and extras {'vocab': <allennlp.data.vocabulary.Vocabulary object at 0x7fef2d2589b0>}
2018-08-24 15:53:38,557 - INFO - allennlp.common.from_params - instantiating class <class 'allennlp.models.reading_comprehension.dialog_qa.DialogQA'> from params {'dropout': 0.3, 'initializer': [], 'marker_embedding_dim': 10, 'num_context_answers': 2, 'phrase_layer': {'bidirectional': True, 'hidden_size': 10, 'input_size': 120, 'num_layers': 1, 'type': 'gru'}, 'residual_encoder': {'bidirectional': True, 'hidden_size': 10, 'input_size': 20, 'num_layers': 1, 'type': 'gru'}, 'span_end_encoder': {'bidirectional': True, 'hidden_size': 10, 'input_size': 40, 'num_layers': 1, 'type': 'gru'}, 'span_start_encoder': {'bidirectional': True, 'hidden_size': 10, 'input_size': 20, 'num_layers': 1, 'type': 'gru'}, 'text_field_embedder': {'token_characters': {'dropout': 0.2, 'embedding': {'embedding_dim': 20, 'num_embeddings': 262}, 'encoder': {'embedding_dim': 20, 'ngram_filter_sizes': [5], 'num_filters': 100, 'type': 'cnn'}, 'type': 'character_encoding'}}} and extras {'vocab': <allennlp.data.vocabulary.Vocabulary object at 0x7fef2d2589b0>}
2018-08-24 15:53:38,557 - INFO - allennlp.common.from_params - instantiating class <class 'allennlp.modules.text_field_embedders.text_field_embedder.TextFieldEmbedder'> from params {'token_characters': {'dropout': 0.2, 'embedding': {'embedding_dim': 20, 'num_embeddings': 262}, 'encoder': {'embedding_dim': 20, 'ngram_filter_sizes': [5], 'num_filters': 100, 'type': 'cnn'}, 'type': 'character_encoding'}} and extras {'vocab': <allennlp.data.vocabulary.Vocabulary object at 0x7fef2d2589b0>}
2018-08-24 15:53:38,557 - INFO - allennlp.common.from_params - instantiating class <class 'allennlp.modules.token_embedders.token_embedder.TokenEmbedder'> from params {'dropout': 0.2, 'embedding': {'embedding_dim': 20, 'num_embeddings': 262}, 'encoder': {'embedding_dim': 20, 'ngram_filter_sizes': [5], 'num_filters': 100, 'type': 'cnn'}, 'type': 'character_encoding'} and extras {'vocab': <allennlp.data.vocabulary.Vocabulary object at 0x7fef2d2589b0>}
2018-08-24 15:53:38,558 - INFO - allennlp.common.from_params - instantiating class <class 'allennlp.modules.seq2vec_encoders.seq2vec_encoder.Seq2VecEncoder'> from params {'embedding_dim': 20, 'ngram_filter_sizes': [5], 'num_filters': 100, 'type': 'cnn'} and extras {}
2018-08-24 15:53:38,558 - INFO - allennlp.common.from_params - instantiating class <class 'allennlp.modules.seq2vec_encoders.cnn_encoder.CnnEncoder'> from params {'embedding_dim': 20, 'ngram_filter_sizes': [5], 'num_filters': 100} and extras {}
2018-08-24 15:53:38,558 - INFO - allennlp.common.registrable - instantiating registered subclass relu of <class 'allennlp.nn.activations.Activation'>
2018-08-24 15:53:38,560 - INFO - allennlp.common.from_params - instantiating class <class 'allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder'> from params {'bidirectional': True, 'hidden_size': 10, 'input_size': 120, 'num_layers': 1, 'type': 'gru'} and extras {'vocab': <allennlp.data.vocabulary.Vocabulary object at 0x7fef2d2589b0>}
2018-08-24 15:53:38,560 - INFO - allennlp.common.from_params - instantiating class <class 'allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder'> from params {'bidirectional': True, 'hidden_size': 10, 'input_size': 20, 'num_layers': 1, 'type': 'gru'} and extras {'vocab': <allennlp.data.vocabulary.Vocabulary object at 0x7fef2d2589b0>}
2018-08-24 15:53:38,561 - INFO - allennlp.common.from_params - instantiating class <class 'allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder'> from params {'bidirectional': True, 'hidden_size': 10, 'input_size': 20, 'num_layers': 1, 'type': 'gru'} and extras {'vocab': <allennlp.data.vocabulary.Vocabulary object at 0x7fef2d2589b0>}
2018-08-24 15:53:38,561 - INFO - allennlp.common.from_params - instantiating class <class 'allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder'> from params {'bidirectional': True, 'hidden_size': 10, 'input_size': 40, 'num_layers': 1, 'type': 'gru'} and extras {'vocab': <allennlp.data.vocabulary.Vocabulary object at 0x7fef2d2589b0>}
2018-08-24 15:53:38,562 - INFO - allennlp.common.registrable - instantiating registered subclass linear of <class 'allennlp.nn.activations.Activation'>
2018-08-24 15:53:38,562 - INFO - allennlp.common.registrable - instantiating registered subclass linear of <class 'allennlp.nn.activations.Activation'>
2018-08-24 15:53:38,563 - INFO - allennlp.data.dataset_readers.reading_comprehension.quac - Reading file at /home/eunsol/Project/quac/allennlp-quac/allennlp/tests/fixtures/data/quac_sample.json
2018-08-24 15:53:38,563 - INFO - allennlp.data.dataset_readers.reading_comprehension.quac - Reading the dataset
2018-08-24 15:53:38,587 - INFO - allennlp.data.dataset_readers.reading_comprehension.quac - Reading file at /home/eunsol/Project/quac/allennlp-quac/allennlp/tests/fixtures/data/quac_sample.json
2018-08-24 15:53:38,588 - INFO - allennlp.data.dataset_readers.reading_comprehension.quac - Reading the dataset
2018-08-24 15:53:38,612 - INFO - allennlp.data.dataset_readers.reading_comprehension.quac - Reading file at /home/eunsol/Project/quac/allennlp-quac/allennlp/tests/fixtures/data/quac_sample.json
2018-08-24 15:53:38,613 - INFO - allennlp.data.dataset_readers.reading_comprehension.quac - Reading the dataset
=============================== warnings summary ===============================
allennlp/tests/models/reading_comprehension/dialog_qa_test.py::DialogQATest::test_forward_pass_runs_correctly
  /home/eunsol/anaconda3/envs/allennlp/lib/python3.6/site-packages/torch/tensor.py:255: UserWarning: non-inplace resize is deprecated
    warnings.warn("non-inplace resize is deprecated")

allennlp/tests/models/reading_comprehension/dialog_qa_test.py::DialogQATest::test_model_can_train_save_and_load
  /home/eunsol/anaconda3/envs/allennlp/lib/python3.6/site-packages/torch/tensor.py:255: UserWarning: non-inplace resize is deprecated
    warnings.warn("non-inplace resize is deprecated")

-- Docs: http://doc.pytest.org/en/latest/warnings.html
=============== 1 failed, 2 passed, 2 warnings in 11.01 seconds ================
