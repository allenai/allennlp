import pytest
from allennlp.data import Tokenizer

from allennlp.data.dataset_readers.huggingface_datasets_reader import HuggingfaceDatasetReader
from allennlp.data.tokenizers import WhitespaceTokenizer


# TODO Add test where we compare huggingface wrapped reader with an explicitly coded dataset
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

    # TODO pab-vmware skip these once MR is ready to check-in
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

    # TODO pab-vmware skip these once MR is ready to check-in
    @pytest.mark.parametrize("dataset", (("swahili"), ("dbpedia_14"), ("trec"), ("emotion")))
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

    # def test_conll2003(self):
    #     instances = list(HuggingfaceDatasetReader("conll2003").read("test"))
    #     print(instances[0])

    # @pytest.mark.skip("Requires implementation of Dict")
    def test_squad(self):
        tokenizer: Tokenizer = WhitespaceTokenizer()
        instances = list(HuggingfaceDatasetReader("squad", tokenizer=tokenizer).read("train"))
        print(instances[0])

    @pytest.mark.parametrize("config", (("default"), ("ptb")))
    def test_sst(self, config):
        instances = list(HuggingfaceDatasetReader("sst", config).read("test"))
        print(instances[0])

    def test_open_web_text(self):
        instances = list(HuggingfaceDatasetReader("openwebtext").read("plain_text"))
        print(instances[0])

    # @pytest.mark.skip("Requires mapping of dict type")
    def test_mocha(self):
        instances = list(HuggingfaceDatasetReader("mocha").read("test"))
        print(instances[0])

    @pytest.mark.skip("Requires implementation of Dict")
    def test_commonsense_qa(self):
        instances = list(HuggingfaceDatasetReader("commonsense_qa").read("test"))
        print(instances[0])

    def test_piqa(self):
        instances = list(HuggingfaceDatasetReader("piqa").read("test"))
        print(instances[0])

    def test_swag(self):
        instances = list(HuggingfaceDatasetReader("swag").read("test"))
        print(instances[0])

    def test_snli(self):
        instances = list(HuggingfaceDatasetReader("snli").read("test"))
        print(instances[0])

    def test_multi_nli(self):
        instances = list(HuggingfaceDatasetReader("multi_nli").read("test"))
        print(instances[0])

    def test_super_glue(self):
        instances = list(HuggingfaceDatasetReader("super_glue").read("test"))
        print(instances[0])

    @pytest.mark.parametrize(
        "config",
        (
            ("cola"),
            ("mnli"),
            ("ax"),
            ("mnli_matched"),
            ("mnli_mismatched"),
            ("mrpc"),
            ("qnli"),
            ("qqp"),
            ("rte"),
            ("sst2"),
            ("stsb"),
            ("wnli"),
        ),
    )
    def test_glue(self, config):
        instances = list(HuggingfaceDatasetReader("glue", config).read("test"))
        print(instances[0])

    def test_drop(self):
        instances = list(HuggingfaceDatasetReader("drop").read("test"))
        print(instances[0])
