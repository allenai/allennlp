from allennlp.data import DatasetReader, Token, Field, Tokenizer
from allennlp.data.fields import TextField, LabelField, ListField
from allennlp.data.instance import Instance
from datasets import load_dataset, DatasetDict, Split, list_datasets
from datasets.features import (
    ClassLabel,
    Sequence,
    Translation,
    TranslationVariableLanguages,
    Value,
    FeatureType,
)
from typing import Iterable, Optional, Dict, List, Union


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
        features: Dict[str, FeatureType] = self.dataset[split].features
        fields: Dict[str, Field] = dict()

        # TODO we need to support all different datasets features described
        # in https://huggingface.co/docs/datasets/features.html
        for feature in features:
            fields_to_be_added: Dict[str, Field] = dict()
            item_field: Field
            field_list: list
            value = features[feature]

            fields_to_be_added = map_Feature(feature, inputs[1], value, self.tokenizer)
            for field_key in fields_to_be_added:
                fields[field_key] = fields_to_be_added[field_key]

        return Instance(fields)


# Feature Mappers - These functions map a FeatureType into Fields
def map_Feature(
    feature: str, entry: Dict, value, tokenizer: Optional[Tokenizer]
) -> Dict[str, Field]:
    fields_to_be_added: Dict[str, Field] = dict()
    if isinstance(value, ClassLabel):
        fields_to_be_added[feature] = map_ClassLabel(feature, entry[feature])
    # datasets Value can be of different types
    elif isinstance(value, Value):
        fields_to_be_added[feature] = map_Value(feature, entry[feature], value, tokenizer)

    elif isinstance(value, Sequence):
        fields_to_be_added = map_Sequence(feature, entry, value, tokenizer)

    elif isinstance(value, Translation):
        fields_to_be_added = map_Translation(feature, entry, value, tokenizer)

    elif isinstance(value, TranslationVariableLanguages):
        fields_to_be_added = map_TranslationVariableLanguages(feature, entry, value, tokenizer)

    else:
        raise ValueError(f"Datasets feature type {type(value)} is not supported yet.")
    return fields_to_be_added


def map_ClassLabel(feature: str, entry: Dict) -> Field:
    field: Field = map_to_Label(feature, entry, skip_indexing=True)
    return field


def map_Value(
    feature: str, item: Value, value, tokenizer: Optional[Tokenizer]
) -> Union[TextField, LabelField]:
    field: Union[TextField, LabelField]
    if value.dtype == "string":
        # datasets.Value[string] maps to TextField
        # If tokenizer is provided we will use it to split it to tokens
        # Else put whole text as a single token
        field = map_String(feature, item, None, tokenizer)

    else:
        field = LabelField(item, label_namespace=feature, skip_indexing=True)
    return field


def map_Sequence(
    feature: str, entry: Dict, value, tokenizer: Optional[Tokenizer]
) -> Dict[str, Field]:
    item_field: Union[LabelField, TextField]
    field_list: List[Union[TextField, LabelField]] = list()
    fields: Dict[str, Field] = dict()
    if isinstance(value.feature, Value):
        for item in entry[feature]:
            # If tokenizer is provided we will use it to split it to tokens
            # Else put whole text as a single token
            item_field = map_Value(feature, item, value.feature, tokenizer)
            field_list.append(item_field)
        if len(field_list) > 0:
            fields[feature] = ListField(field_list)

    # datasets Sequence of strings to ListField of LabelField
    elif isinstance(value.feature, ClassLabel):
        for item in entry[feature]:
            item_field = map_to_Label(feature, item, skip_indexing=True)
            field_list.append(item_field)

        if len(field_list) > 0:
            fields[feature] = ListField(field_list)

    else:
        HuggingfaceDatasetReader.raise_feature_not_supported_value_error(value)

    return fields


def map_Translation(
    feature: str, entry: Dict, value, tokenizer: Optional[Tokenizer]
) -> Dict[str, Field]:
    fields: Dict[str, Field] = dict()
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
            [map_to_Label(feature + "-languages", lang, skip_indexing=False) for lang in langs]
        )
        fields[feature + "-texts"] = ListField(texts)

    else:
        raise ValueError(f"Datasets feature type {type(value)} is not supported yet.")

    return fields


def map_TranslationVariableLanguages(
    feature: str, entry: Dict, value, tokenizer: Optional[Tokenizer]
) -> Dict[str, Field]:
    fields: Dict[str, Field] = dict()
    if value.dtype == "dict":
        input_dict = entry[feature]
        fields[feature + "-language"] = ListField(
            [
                map_to_Label(feature + "-languages", lang, skip_indexing=False)
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


# Value mapper - Maps a single Value
def map_String(feature: str, text: str, value, tokenizer: Optional[Tokenizer]) -> TextField:
    field: TextField
    if tokenizer is not None:
        field = TextField(tokenizer.tokenize(text))
    else:
        field = TextField([Token(text)])
    return field


def map_to_Label(namespace, item, skip_indexing=True) -> LabelField:
    return LabelField(label=item, label_namespace=namespace, skip_indexing=skip_indexing)
