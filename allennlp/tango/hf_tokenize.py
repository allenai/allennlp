"""
*AllenNLP Tango is an experimental API and parts of it might change or disappear
every time we release a new version.*
"""

import copy
import re
from typing import Optional, List, Any, Iterable, Tuple, Dict

import torch
from allennlp.common import cached_transformers
from allennlp.data import Vocabulary
from allennlp.data.fields import TransformerTextField
from allennlp.tango.dataset import DatasetDict
from allennlp.tango.step import Step
from datasets import Dataset


@Step.register("hf_tokenize")
class HuggingfaceTokenize(Step):
    """This step converts strings in the original dataset into `TransformerTextField`s."""

    DETERMINISTIC = True
    VERSION = "001"
    CACHEABLE = True

    def run(  # type: ignore
        self,
        tokenizer_name: str,
        input: DatasetDict,
        fields_to_tokenize: Optional[List[str]] = None,
        add_special_tokens: bool = True,
        max_length: Optional[int] = 512,
        special_tokens_mask: bool = False,
        offset_mapping: bool = False,
    ) -> DatasetDict:
        """
        Reads a dataset and converts all strings in it into `TransformerTextField`.

        * `tokenizer_name` is the name of the tokenizer to use. For example, `"roberta-large"`.
        * `input` is the dataset to transform in this way.
        * By default, this step tokenizes all strings it finds, but if you specify
          `fields_to_tokenize`, it will only tokenize the named fields.
        * `add_special_tokens` specifies whether or not to add special tokens to the tokenized strings.
        * `max_length` is the maximum length the resulting `TransformerTextField` will have.
          If there is too much text, it will be truncated.
        * `special_tokens_mask` specifies whether to add the special token mask as one of the
           tensors in `TransformerTextField`.
        * `offset_mapping` specifies whether to add a mapping from tokens to original string
           offsets to the tensors in `TransformerTextField`.

        This function returns a new dataset with new `TransformerTextField`s.
        """

        tokenizer = cached_transformers.get_tokenizer(tokenizer_name)
        assert tokenizer.pad_token_type_id == 0

        field_names_used = set()

        # find all the strings
        if fields_to_tokenize is None:

            def should_tokenize_field(fname: str) -> bool:
                return True

        else:
            regexes_to_tokenize = [re.compile(r) for r in fields_to_tokenize]

            def should_tokenize_field(fname: str) -> bool:
                for r in regexes_to_tokenize:
                    if r.fullmatch(fname):
                        return True
                return False

        def find_string_objects(o: Any, prefix: str = "") -> Iterable[Tuple[str, str]]:
            prefix = prefix.lstrip(".")
            if isinstance(o, str):
                if should_tokenize_field(prefix):
                    yield prefix, o
            elif isinstance(o, List):
                for i, item in enumerate(o):
                    yield from find_string_objects(item, f"{prefix}.{i}")
            elif isinstance(o, Dict):
                for name, item in o.items():
                    yield from find_string_objects(item, f"{prefix}.{name}")

        strings = []
        for split_name, instances in input.splits.items():
            for instance in instances:
                for name, string in find_string_objects(instance):
                    field_names_used.add(name)
                    strings.append(string)

        for field_name in sorted(field_names_used):
            self.logger.info("Tokenizing field %s", field_name)

        # This thing is so complicated because we want to call `batch_encode_plus` with all
        # the strings at once.
        encoded = tokenizer.batch_encode_plus(
            strings,
            add_special_tokens=add_special_tokens,
            truncation=max_length is not None,
            max_length=max_length,
            return_token_type_ids=True,
            return_attention_mask=False,
            return_special_tokens_mask=special_tokens_mask,
            return_offsets_mapping=offset_mapping,
        )

        # make fields
        string_to_field = {
            s: TransformerTextField(
                torch.tensor(encoded["input_ids"][i], dtype=torch.int32),
                torch.tensor(encoded["token_type_ids"][i], dtype=torch.int32),
                torch.tensor(encoded["attention_mask"][i], dtype=torch.bool)
                if "attention_mask" in encoded
                else None,
                torch.tensor(encoded["special_tokens_mask"][i], dtype=torch.bool)
                if "special_tokens_mask" in encoded
                else None,
                torch.tensor(encoded["offset_mapping"][i], dtype=torch.int32)
                if "offset_mapping" in encoded
                else None,
                tokenizer.pad_token_id,
            )
            for i, s in enumerate(strings)
        }

        def replace_string_objects(o: Any) -> Any:
            if isinstance(o, str):
                try:
                    return string_to_field[o]
                except KeyError:
                    return o
            elif isinstance(o, List) or isinstance(o, Dataset):
                return [replace_string_objects(i) for i in o]
            elif isinstance(o, Dict):
                return {key: replace_string_objects(value) for key, value in o.items()}
            else:
                return o

        new_splits = {
            split_name: replace_string_objects(split_data)
            for split_name, split_data in input.splits.items()
        }

        # make vocab
        if input.vocab is not None:
            vocab = copy.deepcopy(input.vocab)
        else:
            vocab = Vocabulary.empty()

        for name in field_names_used:
            vocab.add_transformer_vocab(tokenizer, name)

        return DatasetDict(new_splits, vocab)
