import typing
from typing import Iterable, Optional

import datasets
from allennlp.data import DatasetReader, Token, Field, Tokenizer
from allennlp.data.fields import TextField, LabelField, ListField
from allennlp.data.instance import Instance
from datasets import load_dataset, DatasetDict, Split, list_datasets
from datasets.features import ClassLabel, Sequence, Translation, TranslationVariableLanguages
from datasets.features import Value


@DatasetReader.register("huggingface-datasets")
class HuggingfaceDatasetReader(DatasetReader):
    """
    Reads instances from the given huggingface supported dataset

    This reader implementation wraps the huggingface datasets package

    Following dataset and configurations have been verified and work with this reader

            Dataset                       Dataset Configuration
            `xnli`                        `ar`
            `xnli`                        `en`
            `xnli`                        `de`
            `xnli`                        `all_languages`
            `glue`                        `cola`
            `glue`                        `mrpc`
            `glue`                        `sst2`
            `glue`                        `qqp`
            `glue`                        `mnli`
            `glue`                        `mnli_matched`
            `universal_dependencies`      `en_lines`
            `universal_dependencies`      `ko_kaist`
            `universal_dependencies`      `af_afribooms`
            `swahili`                     `NA`
            `conll2003`                   `NA`
            `dbpedia_14`                  `NA`
            `trec`                        `NA`
            `emotion`                     `NA`
             Note: universal_dependencies will require you to install `conllu` package separately

    Registered as a `DatasetReader` with name `huggingface-datasets`

    # Parameters

    dataset_name : `str`
        Name of the dataset from huggingface datasets the reader will be used for.
    config_name : `str`, optional (default=`None`)
        Configuration(mandatory for some datasets) of the dataset.
    preload : `bool`, optional (default=`False`)
        If `True` all splits for the dataset is loaded(includes download etc) as part of the initialization,
        otherwise each split is loaded on when `read()` is used for the same for the first time.
    tokenizer : `Tokenizer`, optional (default=`None`)
        If specified is used for tokenization of string and text fields from the dataset.
        This is useful since text in allennlp is dealt with as a series of tokens.
    """

    SUPPORTED_SPLITS = [Split.TRAIN, Split.TEST, Split.VALIDATION]

    def __init__(
        self,
        dataset_name: str = None,
        config_name: Optional[str] = None,
        tokenizer: Optional[Tokenizer] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            manual_distributed_sharding=True,
            manual_multiprocess_sharding=True,
            **kwargs,
        )

        # It would be cleaner to create a separate reader object for diferent dataset
        if dataset_name not in list_datasets():
            raise ValueError(f"Dataset {dataset_name} not available in huggingface datasets")
        self.dataset: DatasetDict = DatasetDict()
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.tokenizer = tokenizer

    def load_dataset_split(self, split: str):
        # TODO add support for datasets.split.NamedSplit
        if split in self.SUPPORTED_SPLITS:
            if self.config_name is not None:
                self.dataset[split] = load_dataset(self.dataset_name, self.config_name, split=split)
            else:
                self.dataset[split] = load_dataset(self.dataset_name, split=split)
        else:
            raise ValueError(
                f"Only default splits:{self.SUPPORTED_SPLITS} are currently supported."
            )

    def _read(self, file_path: str) -> Iterable[Instance]:
        """
        Reads the dataset and converts the entry to AllenNLP friendly instance
        """
        if file_path is None:
            raise ValueError("parameter split cannot be None")

        # If split is not loaded, load the specific split
        if file_path not in self.dataset:
            self.load_dataset_split(file_path)

        # TODO see if use of Dataset.select() is better
        for entry in self.shard_iterable(self.dataset[file_path]):
            yield self.text_to_instance(file_path, entry)

    def raise_feature_not_supported_value_error(value):
        raise ValueError(f"Datasets feature type {type(value)} is not supported yet.")

    def text_to_instance(self, *inputs) -> Instance:
        """
        Takes care of converting dataset entry into AllenNLP friendly instance
        Currently it is implemented in an unseemly catch-up model
        where it converts datasets.features that are required for the supported dataset,
         ideally it would require design where we cleanly deliberate, decide
        map dataset.feature to an allenlp.data.field  and then go ahead with converting it
        Doing that would provide the best chance of providing largest possible coverage with datasets

        Currently this is how datasets.features types are mapped to AllenNLP Fields

        dataset.feature type        allennlp.data.fields
        `ClassLabel`                  `LabelField` in feature name namespace
        `Value.string`                `TextField` with value as Token
        `Value.*`                     `LabelField` with value being label in feature name namespace
        `Sequence.string`             `ListField` of `TextField` with individual string as token
        `Sequence.ClassLabel`         `ListField` of `ClassLabel` in feature name namespace
        `Translation`                 `ListField` of 2 ListField (ClassLabel and TextField)
        `TranslationVariableLanguages`                 `ListField` of 2 ListField (ClassLabel and TextField)
        """

        # features indicate the different information available in each entry from dataset
        # feature types decide what type of information they are
        # e.g. In a Sentiment dataset an entry could have one feature (of type text/string) indicating the text
        # and another indicate the sentiment (of typeint32/ClassLabel)

        split = inputs[0]
        features: typing.List[datasets.features.Feature] = self.dataset[split].features
        fields = dict()

        # TODO we need to support all different datasets features described
        # in https://huggingface.co/docs/datasets/features.html
        for feature in features:
            fields_to_be_added: typing.Dict[str, Field] = dict()
            item_field: Field
            field_list: list
            value = features[feature]

            # datasets ClassLabel maps to LabelField
            if isinstance(value, ClassLabel):
                fields_to_be_added = map_ClassLabel(feature, inputs[1])

            # datasets Value can be of different types
            elif isinstance(value, Value):
                fields_to_be_added = map_Value(feature, inputs[1], value, self.tokenizer)

            elif isinstance(value, Sequence):
                fields_to_be_added = map_Sequence(feature, inputs[1], value, self.tokenizer)

            elif isinstance(value, Translation):
                fields_to_be_added = map_Translation(feature, inputs[1], value, self.tokenizer)

            elif isinstance(value, TranslationVariableLanguages):
                fields_to_be_added = map_TranslationVariableLanguages(
                    feature, inputs[1], value, self.tokenizer
                )

            else:
                raise ValueError(f"Datasets feature type {type(value)} is not supported yet.")

            for field_key in fields_to_be_added:
                fields[field_key] = fields_to_be_added[field_key]

        return Instance(fields)


def map_ClassLabel(feature: str, entry: typing.Dict) -> typing.Dict[str, Field]:
    fields: typing.Dict[str, Field] = dict()
    fields[feature] = LabelField(entry[feature], label_namespace=feature, skip_indexing=True)
    return fields


def map_Value(
    feature: str, entry: typing.Dict, value, tokenizer: Optional[Tokenizer]
) -> typing.Dict[str, Field]:
    fields: typing.Dict[str, Field] = dict()
    if value.dtype == "string":
        # datasets.Value[string] maps to TextField
        # If tokenizer is provided we will use it to split it to tokens
        # Else put whole text as a single token
        if tokenizer is not None:
            fields[feature] = TextField(tokenizer.tokenize(entry[feature]))

        else:
            fields[feature] = TextField([Token(entry[feature])])

    else:
        fields[feature] = LabelField(entry[feature], label_namespace=feature, skip_indexing=True)
    return fields


def map_Sequence(
    feature: str, entry: typing.Dict, value, tokenizer: Optional[Tokenizer]
) -> typing.Dict[str, Field]:
    item_field: typing.Union[LabelField, TextField]
    fields: typing.Dict[str, Field] = dict()
    if hasattr(value.feature, "dtype") and value.feature.dtype == "string":
        field_list2: typing.List[TextField] = list()
        for item in entry[feature]:
            # If tokenizer is provided we will use it to split it to tokens
            # Else put whole text as a single token
            tokens: typing.List[Token]
            if tokenizer is not None:
                tokens = tokenizer.tokenize(item)

            else:
                tokens = [Token(item)]

            item_field = TextField(tokens)
            field_list2.append(item_field)

        fields[feature] = ListField(field_list2)

    # datasets Sequence of strings to ListField of LabelField
    elif isinstance(value.feature, ClassLabel):
        field_list = list()
        for item in entry[feature]:
            item_field = LabelField(label=item, label_namespace=feature, skip_indexing=True)
            field_list.append(item_field)

        fields[feature] = ListField(field_list)

    else:
        HuggingfaceDatasetReader.raise_feature_not_supported_value_error(value)

    return fields


def map_Translation(
    feature: str, entry: typing.Dict, value, tokenizer: Optional[Tokenizer]
) -> typing.Dict[str, Field]:
    fields: typing.Dict[str, Field] = dict()
    if value.dtype == "dict":
        input_dict = entry[feature]
        langs = list(input_dict.keys())
        texts = list()
        for lang in langs:
            if tokenizer is not None:
                tokens = tokenizer.tokenize(input_dict[lang])

            else:
                tokens = [Token(input_dict[lang])]
            texts.append(TextField(tokens))

        fields[feature + "-languages"] = ListField(
            [LabelField(lang, label_namespace="languages") for lang in langs]
        )
        fields[feature + "-texts"] = ListField(texts)

    else:
        raise ValueError(f"Datasets feature type {type(value)} is not supported yet.")

    return fields


def map_TranslationVariableLanguages(
    feature: str, entry: typing.Dict, value, tokenizer: Optional[Tokenizer]
) -> typing.Dict[str, Field]:
    fields: typing.Dict[str, Field] = dict()
    if value.dtype == "dict":
        input_dict = entry[feature]
        fields[feature + "-language"] = ListField(
            [
                LabelField(lang, label_namespace=feature + "-language")
                for lang in input_dict["language"]
            ]
        )

        if tokenizer is not None:
            fields[feature + "-translation"] = ListField(
                [TextField(tokenizer.tokenize(text)) for text in input_dict["translation"]]
            )
        else:
            fields[feature + "-translation"] = ListField(
                [TextField([Token(text)]) for text in input_dict["translation"]]
            )

    else:
        raise ValueError(f"Datasets feature type {type(value)} is not supported yet.")

    return fields
