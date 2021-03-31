import pytest
from allennlp.data import Tokenizer

from allennlp.data.dataset_readers.huggingface_datasets_reader import HuggingfaceDatasetReader
from allennlp.data.tokenizers import WhitespaceTokenizer


# TODO Add test where we compare huggingface wrapped reader with an explicitly built dataset
# TODO pab-vmware/Abhishek-P Add test where we load conll2003 and test it
#  the way tested for conll2003 specific reader
class HuggingfaceDatasetReaderTest:

    """
    Test read for some lightweight datasets
    """

    @pytest.mark.parametrize(
        "dataset, config, split",
        (("glue", "cola", "train"), ("glue", "cola", "test")),
    )
    def test_read(self, dataset, config, split):
        huggingface_reader = HuggingfaceDatasetReader(dataset_name=dataset, config_name=config)
        instances = list(huggingface_reader.read(split))
        # Confirm instance were made for all rows
        assert len(instances) == len(huggingface_reader.dataset[split])

        entry = huggingface_reader.dataset[split][0]
        instance = instances[0]

        # Confirm all features were mapped
        assert len(instance.fields) == len(entry)

    def test_read_unsupported_sequence_nesting(self):
        dataset = "diplomacy_detection"
        split = "train"
        huggingface_reader = HuggingfaceDatasetReader(dataset_name=dataset)
        with pytest.raises(ValueError):
            next(huggingface_reader.read(split))

    def test_read_with_tokenizer(self):
        dataset = "glue"
        config = "cola"
        split = "train"
        tokenizer: Tokenizer = WhitespaceTokenizer()
        huggingface_reader = HuggingfaceDatasetReader(
            dataset_name=dataset, config_name=config, tokenizer=tokenizer
        )
        instances = list(huggingface_reader.read(split))
        # Confirm instance were made for all rows
        assert len(instances) == len(huggingface_reader.dataset[split])

        entry = huggingface_reader.dataset[split][0]
        instance = instances[0]

        # Confirm all features were mapped
        assert len(instance.fields) == len(entry)

        # Confirm it was tokenized
        assert len(instance["sentence"]) > 1

    def test_read_without_config(self):
        dataset = "urdu_fake_news"
        split = "train"
        huggingface_reader = HuggingfaceDatasetReader(dataset_name=dataset)
        instances = list(huggingface_reader.read(split))
        # Confirm instance were made for all rows
        assert len(instances) == len(huggingface_reader.dataset[split])

        entry = huggingface_reader.dataset[split][0]
        instance = instances[0]

        # Confirm all features were mapped
        assert len(instance.fields) == len(entry)

    def test_read_with_preload(self):
        dataset = "glue"
        config = "cola"
        split = "train"
        tokenizer: Tokenizer = WhitespaceTokenizer()
        huggingface_reader = HuggingfaceDatasetReader(
            dataset_name=dataset, config_name=config, tokenizer=tokenizer, preload=True
        )
        instances = list(huggingface_reader.read(split))
        # Confirm instance were made for all rows
        assert len(instances) == len(huggingface_reader.dataset[split])

        entry = huggingface_reader.dataset[split][0]
        instance = instances[0]

        # Confirm all features were mapped
        assert len(instance.fields) == len(entry)

        # Confirm it was tokenized
        assert len(instance["sentence"]) > 1

    """
    Test mapping of the datasets.feature.Translation and datasets.feature.TranslationVariableLanguages
    """

    def test_read_xnli_all_languages(self):
        dataset = "xnli"
        config = "all_languages"
        split = "validation"
        huggingface_reader = HuggingfaceDatasetReader(dataset_name=dataset, config_name=config)
        instances = list(huggingface_reader.read(split))
        # Confirm instance were made for all rows
        assert len(instances) == len(huggingface_reader.dataset[split])
        instance = instances[0]
        # We are splitting datasets.features.Translation and
        # datasets.features.TranslationVariableLanguages into two fields each
        # For XNLI that means 3 fields become 5
        assert len(instance.fields) == 5

    def test_non_supported_feature(self):
        dataset = "pubmed_qa"
        config = "pqa_labeled"
        split = "train"
        with pytest.raises(ValueError):
            next(HuggingfaceDatasetReader(dataset_name=dataset, config_name=config).read(split))

    def test_non_available_dataset(self):
        with pytest.raises(ValueError):
            HuggingfaceDatasetReader(dataset_name="surely-such-a-dataset-does-not-exist")

    @pytest.mark.parametrize("split", (None, "surely-such-a-split-does-not-exist"))
    def test_read_with_invalid_split(self, split):
        with pytest.raises(ValueError):
            next(HuggingfaceDatasetReader(dataset_name="glue", config_name="cola").read(split))

    """
    Test to help validate for the known supported datasets
    Skipped by default, enable when required
    """

    @pytest.mark.skip()
    @pytest.mark.parametrize(
        "dataset, config, split",
        (
            ("xnli", "ar", "train"),
            ("xnli", "en", "train"),
            ("xnli", "de", "train"),
            ("glue", "mrpc", "train"),
            ("glue", "sst2", "train"),
            ("glue", "qqp", "train"),
            ("glue", "mnli", "train"),
            ("glue", "mnli_matched", "validation"),
            ("universal_dependencies", "en_lines", "train"),
            ("universal_dependencies", "ko_kaist", "train"),
            ("universal_dependencies", "af_afribooms", "train"),
        ),
    )
    def test_read_known_supported_datasets_with_config(self, dataset, config, split):
        huggingface_reader = HuggingfaceDatasetReader(dataset_name=dataset, config_name=config)
        instances = list(huggingface_reader.read(split))
        # Confirm instance were made for all rows
        assert len(instances) == len(huggingface_reader.dataset[split])

        entry = huggingface_reader.dataset[split][0]
        instance = instances[0]

        # Confirm all features were mapped
        assert len(instance.fields) == len(entry)

    """
        Test to help validate for the known supported datasets without config
        Skipped by default, enable when required
    """

    @pytest.mark.skip()
    @pytest.mark.parametrize(
        "dataset", (("swahili"), ("conll2003"), ("dbpedia_14"), ("trec"), ("emotion"))
    )
    def test_read_known_supported_datasets_without_config(self, dataset):
        split = "train"
        huggingface_reader = HuggingfaceDatasetReader(dataset_name=dataset)
        instances = list(huggingface_reader.read(split))
        # Confirm instance were made for all rows
        assert len(instances) == len(huggingface_reader.dataset[split])

        entry = huggingface_reader.dataset[split][0]
        instance = instances[0]

        # Confirm all features were mapped
        assert len(instance.fields) == len(entry)
