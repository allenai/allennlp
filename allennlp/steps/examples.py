import copy
import dataclasses
import logging
import random
import re
from math import floor, ceil
from typing import (
    Set,
    Optional,
    Union,
    List,
    Dict,
    Any,
    Tuple,
    Iterable,
    Iterator,
    Sequence,
)

import datasets
import more_itertools
import torch.optim
from datasets import Dataset

from allennlp.common import cached_transformers
from allennlp.data import (
    DataLoader,
    Vocabulary,
    BatchSampler,
    Instance,
    allennlp_collate,
    TensorDict,
)
from allennlp.data.fields import LabelField, ListField, IndexField
from allennlp.data.fields.transformer_text_field import TransformerTextField
from allennlp.models import Model
from allennlp.nn.util import move_to_device
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
    """This step converts strings in the original dataset into `TransformerTextField`s."""

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


@Step.register("piqa_instances")
class PiqaInstances(Step):
    DETERMINISTIC = True
    VERSION = "001"
    CACHEABLE = True

    def run(
        self,
        tokenizer_name: str,
        max_length: int = 512,
    ) -> AllenNlpDataset:
        tokenizer = cached_transformers.get_tokenizer(tokenizer_name)
        assert tokenizer.pad_token_type_id == 0

        dataset = {
            split_name: [
                {
                    "correct_alternative": LabelField(instance["label"]),
                    "alternatives": [
                        (instance["goal"], instance["sol1"]),
                        (instance["goal"], instance["sol2"]),
                    ],
                }
                for instance in instances
            ]
            for split_name, instances in datasets.load_dataset("piqa").items()
        }

        # This thing is so complicated because we want to call `batch_encode_plus` with all
        # the strings at once.
        tokenized = {
            split_name: tokenizer.batch_encode_plus(
                [alternative for instance in instances for alternative in instance["alternatives"]],
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
                return_token_type_ids=True,
                return_attention_mask=False,
            )
            for split_name, instances in dataset.items()
        }

        result = {}
        for split_name, instances in dataset.items():
            tokenized_alts = tokenized["split_name"]
            results_per_split = []
            for i, instance in enumerate(instances):
                alts = ListField(
                    [
                        TransformerTextField(
                            torch.tensor(tokenized_alts["input_ids"][alt_index], dtype=torch.int32),
                            torch.tensor(
                                tokenized_alts["token_type_ids"][alt_index], dtype=torch.int32
                            ),
                            torch.tensor(
                                tokenized_alts["attention_mask"][alt_index], dtype=torch.bool
                            ),
                        )
                        for alt_index in [2 * i, 2 * i + 1]
                    ]
                )
                label = IndexField(instance["label"], alts)
                results_per_split.append(
                    Instance({"alternatives": alts, "correct_alternative": label})
                )
            result[split_name] = results_per_split

        # make vocab
        vocab = Vocabulary.empty()
        vocab.add_transformer_vocab(tokenizer, "tokens")

        return AllenNlpDataset(result, vocab)


class TangoDataLoader:
    def num_batches_per_epoch(self) -> Optional[int]:
        """If the dataloader produces epochs of similar length, this is how you get the length."""
        raise NotImplementedError()

    def __iter__(self) -> Iterator[TensorDict]:
        raise NotImplementedError()

    def __len__(self) -> Optional[int]:
        logging.warning(
            "This function is deprecated because it's unclear which length you get back. Please call "
            "TangoDataLoader.num_batches_per_epoch() instead."
        )
        return self.num_batches_per_epoch()


@Step.register("make_data_loaders")
class MakeDataLoadersStep(Step):
    DETERMINISTIC = True
    VERSION = "001"
    CACHEABLE = False

    class BatchSizeDataLoader(TangoDataLoader):
        def __init__(
            self,
            instances: Sequence[Instance],
            batch_size: int,
            drop_last: bool = False,
            shuffle: bool = True,
        ):
            self.instances = instances
            self.batch_size = batch_size
            self.drop_last = drop_last
            self.shuffle = shuffle

        def num_batches_per_epoch(self) -> Optional[int]:
            batch_count = len(self.instances) / self.batch_size
            if self.drop_last:
                return floor(batch_count)
            else:
                return ceil(batch_count)

        def __iter__(self) -> Iterator[TensorDict]:
            if self.shuffle:
                instances = list(
                    self.instances
                )  # make a new list pointing to the same instance objects
                random.shuffle(instances)
            else:
                instances = self.instances

            for batch in more_itertools.chunked(instances, self.batch_size):
                if not self.drop_last or len(batch) >= self.batch_size:
                    yield allennlp_collate(batch)

    class SamplerDataLoader(TangoDataLoader):
        def __init__(self, instances: Sequence[Instance], batch_sampler: BatchSampler):
            self.instances = instances
            self.batch_sampler = batch_sampler

        def num_batches_per_epoch(self) -> Optional[int]:
            return self.batch_sampler.get_num_batches(self.instances)

        def __iter__(self) -> Iterator[TensorDict]:
            for batch_indices in self.batch_sampler.get_batch_indices(self.instances):
                yield allennlp_collate([self.instances[i] for i in batch_indices])

    class BatchesPerEpochDataLoader(TangoDataLoader):
        def __init__(self, inner: TangoDataLoader, batches_per_epoch: int):
            self.inner = inner
            self.iter = iter(inner)
            self.batches_per_epoch = batches_per_epoch

        def num_batches_per_epoch(self) -> Optional[int]:
            return self.batches_per_epoch

        def __iter__(self) -> Iterator[TensorDict]:
            batches_yielded = 0
            while batches_yielded < self.batches_per_epoch:
                try:
                    yield next(self.iter)
                    batches_yielded += 1
                except StopIteration:
                    self.iter = iter(self.inner)

    def run(
        self,
        dataset: AllenNlpDataset,
        batch_size: int = None,
        drop_last: bool = False,
        shuffle: bool = False,
        batch_sampler: Optional[BatchSampler] = None,
        batches_per_epoch: Optional[int] = None,
    ) -> Dict[str, TangoDataLoader]:
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        if batch_sampler is not None:
            if batch_size is not None:
                raise ValueError("batch_sampler option is mutually exclusive with batch_size.")

            if drop_last:
                raise ValueError("batch_sampler option is mutually exclusive with drop_last.")

            if shuffle:
                raise ValueError("batch_sampler option is mutually exclusive with shuffle.")
        elif batch_size is None:
            raise ValueError("batch_size is required when batch_sampler is not supplied.")

        if batches_per_epoch is not None and batches_per_epoch < 1:
            raise ValueError("batches_per_epoch must be at least 1.")

        split_to_loader = {}
        for split_name, instances in dataset.splits.items():
            if batch_sampler is not None:
                loader = self.SamplerDataLoader(instances, batch_sampler)
            else:
                assert batch_size is not None
                loader = self.BatchSizeDataLoader(instances, batch_size, drop_last, shuffle)
            if batches_per_epoch is not None:
                loader = self.BatchesPerEpochDataLoader(loader, batches_per_epoch)
            split_to_loader[split_name] = loader

        return split_to_loader


@Step.register("training")
class TrainingStep(Step):
    DETERMINISTIC = True
    VERSION = "001"

    # TODO: distributed training
    # TODO: recovery of failed jobs

    class DataLoaderAdapter(DataLoader):
        """Adapts a TangoDataLoader to an old-school AllenNLP DataLoader."""

        def __init__(self, tango_data_loader: TangoDataLoader):
            self.tango_data_loader = tango_data_loader
            self.target_device: Optional[torch.device] = None

        def __len__(self) -> int:
            return self.tango_data_loader.num_batches_per_epoch()

        def __iter__(self) -> Iterator[TensorDict]:
            if self.target_device is None:
                return iter(self.tango_data_loader)
            else:
                for batch in iter(self.tango_data_loader):
                    yield move_to_device(batch, self.target_device)

        def iter_instances(self) -> Iterator[Instance]:
            raise NotImplementedError()

        def index_with(self, vocab: Vocabulary) -> None:
            raise NotImplementedError()

        def set_target_device(self, device: torch.device) -> None:
            self.target_device = device

    # Development notes:
    #
    # This is not taking a cuda_device. We autodetect those. If you don't want to run with the GPU, set
    # the CUDA_DEVICES environment variable to be empty.
    #
    # This is adaptering so we can use the original trainer. But the original trainer API is insane. You
    # instantiate the object, and then you can call exactly one method on it (.train()), and you can
    # call it exactly once. If you do anything else crazy things happen. We should replace the trainer API
    # entirely and transplant the logic from the .train() method directly into the step's .run() method.
    # If we do want to have a separate Trainer object, it should take data loaders and models in the .train()
    # method, not in __init__(), and allow multiple calls to that method (even multiple concurrent ones). That
    # would be a sane API.

    def run(
        self,
        model: Model,
        data_loaders: Dict[str, TangoDataLoader],
        validation_data_loaders: Optional[Dict[str, TangoDataLoader]],
        training_split: str,
        validation_split: Optional[str],
        optimizer: torch.optim.Optimizer,
        patience: Optional[int] = None,
        validation_metric: Union[str, List[str]] = "-loss",
        num_epochs: int = 20,
        checkpointer: Checkpointer = None,
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
        if validation_data_loaders is None:
            validation_data_loaders = data_loaders
        if validation_split is None:
            validation_loader = None
        else:
            validation_loader = self.DataLoaderAdapter(validation_data_loaders[validation_split])

        trainer = GradientDescentTrainer(
            model,
            optimizer=optimizer,
            data_loader=self.DataLoaderAdapter(data_loaders[training_split]),
            patience=patience,
            validation_metric=validation_metric,
            validation_data_loader=validation_loader,
            num_epochs=num_epochs,
            serialization_dir=None,
            checkpointer=checkpointer,
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
        trainer.train()

        return model
