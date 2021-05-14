import copy
import dataclasses
import logging
import re
from typing import Set, Optional, Union, List, Dict, Any, Tuple, Iterable

import datasets
import torch.optim
from datasets import Dataset
from transformers import AutoTokenizer

from allennlp.data import DataLoader, Vocabulary
from allennlp.data.fields.transformer_text_field import TransformerTextField
from allennlp.models import Model
from allennlp.steps.dataset import AllenNlpDataset
from allennlp.steps.step import Step
from allennlp.training import Checkpointer, TrainerCallback, GradientDescentTrainer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.momentum_schedulers import MomentumScheduler
from allennlp.training.moving_average import MovingAverage


@Step.register("hf_dataset")
class HuggingfaceDataset(Step):
    DETERMINISTIC = True
    VERSION = "001"
    CACHEABLE = False  # These are already cached by huggingface.

    def run(self, dataset_name: str) -> AllenNlpDataset:
        return AllenNlpDataset(datasets.load_dataset(dataset_name), None, {"source": "huggingface"})


@Step.register("text_only")
class TextOnlyDataset(Step):
    DETERMINISTIC = True

    def run(self, input: AllenNlpDataset, fields_to_keep: Set[str]) -> AllenNlpDataset:
        return dataclasses.replace(
            input,
            splits={
                split_name: [
                    {"text": field_value}
                    for instance in split
                    for field_name, field_value in instance.items()
                    if field_name in fields_to_keep
                ]
                for split_name, split in input.splits.items()
            },
        )


@Step.register("hf_tokenizer")
class Tokenize(Step):
    DETERMINISTIC = True
    VERSION = "001"
    CACHEABLE = True

    def run(
        self,
        tokenizer_name: str,
        input: AllenNlpDataset,
        fields_to_tokenize: Optional[List[str]] = None,
        add_special_tokens: bool = True,
        max_length: Optional[int] = 512,
        special_tokens_mask: bool = False,
        offset_mapping: bool = False,
    ) -> AllenNlpDataset:
        tokenizer = cached_transformers.get_tokenizer(tokenizer_name)
        assert tokenizer.pad_token_type_id == 0

        field_names_used = set()

        # find all the strings
        if fields_to_tokenize is None:

            def should_tokenize_field(_: str) -> bool:
                return True

        else:
            regexes_to_tokenize = [re.compile(r) for r in fields_to_tokenize]

            def should_tokenize_field(field_name: str) -> bool:
                for r in regexes_to_tokenize:
                    if r.fullmatch(field_name):
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
            logging.info("Tokenizing field %s", field_name)

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

        return AllenNlpDataset(new_splits, vocab)


@Step.register("training")
class TrainingStep(Step):
    DETERMINISTIC = True
    VERSION = "001"

    # TODO: distributed training
    # TODO: recovery of failed jobs

    def run(
        self,
        model: Model,
        dataset: AllenNlpDataset,
        data_loader: DataLoader,
        validation_data_loader: Optional[DataLoader],
        training_split: str,
        validation_split: Optional[str],
        optimizer: torch.optim.Optimizer,
        patience: Optional[int] = None,
        validation_metric: Union[str, List[str]] = "-loss",
        num_epochs: int = 20,
        checkpointer: Checkpointer = None,
        cuda_device: Optional[Union[int, torch.device]] = None,
        grad_norm: Optional[float] = None,
        grad_clipping: Optional[float] = None,
        learning_rate_scheduler: Optional[LearningRateScheduler] = None,
        momentum_scheduler: Optional[MomentumScheduler] = None,
        moving_average: Optional[MovingAverage] = None,
        callbacks: List[TrainerCallback] = None,
        num_gradient_accumulation_steps: int = 1,
        use_amp: bool = False,
        enable_default_callbacks: bool = True,
        run_sanity_checks: bool = True,
    ) -> Model:

        trainer = GradientDescentTrainer(
            model,
            optimizer=optimizer,
            data_loader=None,  # dataloader
            patience=patience,
            validation_metric=validation_metric,
            validation_data_loader=None,  # validation dataloader
            num_epochs=num_epochs,
            serialization_dir=None,  # serialization dir
            checkpointer=checkpointer,
            cuda_device=cuda_device,
            grad_norm=grad_norm,
            grad_clipping=grad_clipping,
            learning_rate_scheduler=learning_rate_scheduler,
            momentum_scheduler=momentum_scheduler,
            moving_average=moving_average,
            callbacks=callbacks,
            num_gradient_accumulation_steps=num_gradient_accumulation_steps,
            use_amp=use_amp,
            enable_default_callbacks=enable_default_callbacks,
            run_sanity_checks=run_sanity_checks,
        )
