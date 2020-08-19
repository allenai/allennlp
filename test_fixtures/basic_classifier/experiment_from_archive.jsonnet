local COMMON = import 'common.jsonnet';

{
    "dataset_reader": COMMON['dataset_reader'],
    "datasets_for_vocab_creation": ["train"],
    "train_data_path": "test_fixtures/data/text_classification_json/ag_news_corpus_fake_sentiment_labels.jsonl",
    "validation_data_path": "test_fixtures/data/text_classification_json/ag_news_corpus_fake_sentiment_labels.jsonl",
    "model": {
        "type": "from_archive",
        "archive_file": "test_fixtures/basic_classifier/serialization/model.tar.gz",
    },
    "data_loader": COMMON['data_loader'],
    "trainer": COMMON['trainer'],
}
