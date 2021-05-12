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

    Registered as a `DatasetReader` with name `huggingface-datasets`

    # Parameters
    dataset_name : `str`
        Name of the dataset from huggingface datasets the reader will be used for.
    config_name : `str`, optional (default=`None`)
        Configuration(mandatory for some datasets) of the dataset.
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

        # It would be cleaner to create a separate reader object for each different dataset
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
        dataset_split = self.dataset[file_path]
        for index in self.shard_iterable(range(len(dataset_split))):
            yield self.text_to_instance(file_path, dataset_split[index])

    def raise_feature_not_supported_value_error(value):
        raise ValueError(f"Datasets feature type {type(value)} is not supported yet.")

    def text_to_instance(self, *inputs) -> Instance:
        """
        Takes care of converting dataset entry into AllenNLP friendly instance

        Currently this is how datasets.features types are mapped to AllenNLP Fields

        dataset.feature type           allennlp.data.fields
        `ClassLabel`                   `LabelField` in feature name namespace
        `Value.string`                 `TextField` with value as Token
        `Value.*`                      `LabelField` with value being label in feature name namespace
        `Translation`                  `ListField` of 2 ListField (ClassLabel and TextField)
        `TranslationVariableLanguages` `ListField` of 2 ListField (ClassLabel and TextField)
        `Sequence`                     `ListField` of sub-types
        """

        # features indicate the different information available in each entry from dataset
        # feature types decide what type of information they are
        # e.g. In a Sentiment dataset an entry could have one feature (of type text/string) indicating the text
        # and another indicate the sentiment (of type int32/ClassLabel)

        split = inputs[0]
        features: Dict[str, FeatureType] = self.dataset[split].features
        fields: Dict[str, Field] = dict()

        # TODO we need to support all different datasets features described
        # in https://huggingface.co/docs/datasets/features.html
        for feature_name in features:
            item_field: Field
            field_list: list
            feature_type = features[feature_name]

            fields_to_be_added = _map_Feature(feature_name, inputs[1], feature_type, self.tokenizer)
            for field_key in fields_to_be_added:
                fields[field_key] = fields_to_be_added[field_key]

        return Instance(fields)


# Feature Mappers - These functions map a FeatureType into Fields
def _map_Feature(
    feature_name: str, entry: Dict, feature_type, tokenizer: Optional[Tokenizer]
) -> Dict[str, Field]:
    fields_to_be_added: Dict[str, Field] = dict()
    if isinstance(feature_type, ClassLabel):
        fields_to_be_added[feature_name] = _map_ClassLabel(feature_name, entry[feature_name])
    # datasets Value can be of different types
    elif isinstance(feature_type, Value):
        fields_to_be_added[feature_name] = _map_Value(feature_name, entry[feature_name], feature_type, tokenizer)

    elif isinstance(feature_type, Sequence):
        fields_to_be_added[feature_name] = _map_Sequence(feature_name, entry, feature_type.feature, tokenizer)

    elif isinstance(feature_type, Translation):
        fields_to_be_added = _map_Translation(feature_name, entry[feature_name], feature_type, tokenizer)

    elif isinstance(feature_type, TranslationVariableLanguages):
        fields_to_be_added = _map_TranslationVariableLanguages(feature_name, entry[feature_name], feature_type, tokenizer)

    else:
        raise ValueError(f"Datasets feature type {type(feature_type)} is not supported yet.")
    return fields_to_be_added


def _map_ClassLabel(feature_name: str, value: ClassLabel) -> Field:
    field: Field = _map_to_Label(feature_name, value, skip_indexing=True)
    return field


def _map_Value(
    feature_name: str, value: Value, feature_type, tokenizer: Optional[Tokenizer]
) -> Union[TextField, LabelField]:
    field: Union[TextField, LabelField]
    if feature_type.dtype == "string":
        # datasets.Value[string] maps to TextField
        # If tokenizer is provided we will use it to split it to tokens
        # Else put whole text as a single token
        field = _map_String(value, tokenizer)

    else:
        field = LabelField(value, label_namespace=feature_name, skip_indexing=True)
    return field

def _map_Sequence(feature_name, value:Sequence, item_feature_type, tokenizer:Optional[Tokenizer]) -> Field:
    field_list: List[Field] = list()
    field: ListField = None
    if isinstance(item_feature_type, Value):
        for item in value:
            # If tokenizer is provided we will use it to split it to tokens
            # Else put whole text as a single token
            item_field = _map_Value(value.feature, item, item.value, tokenizer)
            field_list.append(item_field)
        if len(field_list) > 0:
             field = ListField(field_list)

    # datasets Sequence of strings to ListField of LabelField
    elif isinstance(item_feature_type, ClassLabel):
        for item in value:
            item_field = _map_to_Label(value.feature, item, skip_indexing=True)
            field_list.append(item_field)

        if len(field_list) > 0:
            field = ListField(field_list)

    elif isinstance(item_feature_type, Sequence):
        for item in value:
            item_field = _map_Sequence(value.feature, item, tokenizer)
            field_list.append(item_field)

        if len(field_list) > 0:
            field = ListField(field_list)

    else:
        HuggingfaceDatasetReader.raise_feature_not_supported_value_error(feature_name)

    return field

def _map_Translation(
    feature_name: str, value: Translation, feature_type, tokenizer: Optional[Tokenizer]
) -> Dict[str, Field]:
    fields: Dict[str, Field] = dict()
    if feature_type.dtype == "dict":
        input_dict = value
        langs = list(input_dict.keys())
        texts = list()
        for lang in langs:
            if tokenizer is not None:
                tokens = tokenizer.tokenize(input_dict[lang])

            else:
                tokens = [Token(input_dict[lang])]
            texts.append(TextField(tokens))

        fields[feature_name + "-languages"] = ListField(
            [_map_to_Label(feature_name + "-languages", lang, skip_indexing=False) for lang in langs]
        )
        fields[feature_name + "-texts"] = ListField(texts)

    else:
        raise ValueError(f"Datasets feature type {type(feature_type)} is not supported yet.")

    return fields


def _map_TranslationVariableLanguages(
    feature_name: str, value: TranslationVariableLanguages, feature_type, tokenizer: Optional[Tokenizer]
) -> Dict[str, Field]:
    fields: Dict[str, Field] = dict()
    if feature_type.dtype == "dict":
        input_dict = value
        fields[feature_name + "-language"] = ListField(
            [
                _map_to_Label(feature_name + "-languages", lang, skip_indexing=False)
                for lang in input_dict["language"]
            ]
        )

        if tokenizer is not None:
            fields[feature_name + "-translation"] = ListField(
                [TextField(tokenizer.tokenize(text)) for text in input_dict["translation"]]
            )
        else:
            fields[feature_name + "-translation"] = ListField(
                [TextField([Token(text)]) for text in input_dict["translation"]]
            )

    else:
        raise ValueError(f"Datasets feature type {type(value)} is not supported yet.")

    return fields


# value mapper - Maps a single text value to TextField
def _map_String(text: str, tokenizer: Optional[Tokenizer]) -> TextField:
    field: TextField
    if tokenizer is not None:
        field = TextField(tokenizer.tokenize(text))
    else:
        field = TextField([Token(text)])
    return field


# value mapper - Maps a single value to a LabelField
def _map_to_Label(namespace, item, skip_indexing=True) -> LabelField:
    return LabelField(label=item, label_namespace=namespace, skip_indexing=skip_indexing)
