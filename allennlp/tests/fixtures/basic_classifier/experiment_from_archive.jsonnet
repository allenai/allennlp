local COMMON = import 'common.jsonnet';

{
    "dataset_reader": COMMON['dataset_reader'],
    "datasets_for_vocab_creation": ["train"],
    "train_data_path": "allennlp/tests/fixtures/data/text_classification_json/ag_news_corpus_fake_sentiment_labels.jsonl",
    "validation_data_path": "allennlp/tests/fixtures/data/text_classification_json/ag_news_corpus_fake_sentiment_labels.jsonl",
    "model": {
        "type": "from_archive",
        "archive_file": "allennlp/tests/fixtures/basic_classifier/serialization/model.tar.gz",
    },
    "iterator": COMMON['iterator'],
    "trainer": COMMON['trainer'],
}
