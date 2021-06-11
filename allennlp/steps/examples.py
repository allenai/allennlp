import copy
import dataclasses
import logging
import re
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
)

import datasets
import torch.optim
from torch import cuda
from datasets import Dataset

from allennlp.common import cached_transformers, Lazy, Tqdm
from allennlp.common.checks import check_for_gpu
from allennlp.common.util import log_frozen_and_tunable_parameter_names, sanitize
from allennlp.data import (
    DataLoader,
    Vocabulary,
    Instance,
    TensorDict,
    Field,
)
from allennlp.data.fields import ListField, IndexField
from allennlp.data.fields.transformer_text_field import TransformerTextField
from allennlp.models import Model
from allennlp.nn.util import move_to_device
from allennlp.steps.dataloader import TangoDataLoader, MaxBatchesDataLoader, BatchSizeDataLoader, DataLoaderAdapter
from allennlp.steps.dataset import AllenNlpDataset
from allennlp.steps.format import TorchFormat
from allennlp.steps.step import Step
from allennlp.training import Checkpointer, TrainerCallback, GradientDescentTrainer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.momentum_schedulers import MomentumScheduler
from allennlp.training.moving_average import MovingAverage
from allennlp.training.optimizers import Optimizer


logger = logging.getLogger(__name__)


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
    VERSION = "002"
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
                    "correct_alternative": instance["label"],
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
            tokenized_alts = tokenized[split_name]
            results_per_split = []
            for i, instance in enumerate(instances):
                alts = ListField(
                    [
                        TransformerTextField(
                            torch.tensor(tokenized_alts["input_ids"][alt_index], dtype=torch.int32),
                            torch.tensor(
                                tokenized_alts["token_type_ids"][alt_index], dtype=torch.int32
                            ),
                        )
                        for alt_index in [2 * i, 2 * i + 1]
                    ]
                )
                fields: Dict[str, Field] = {"alternatives": alts}
                if instance["correct_alternative"] >= 0:
                    fields["correct_alternative"] = IndexField(
                        instance["correct_alternative"], alts
                    )
                results_per_split.append(Instance(fields))
            result[split_name] = results_per_split

        # make vocab
        vocab = Vocabulary.empty()
        vocab.add_transformer_vocab(tokenizer, "tokens")

        return AllenNlpDataset(result, vocab)


@Step.register("training")
class TrainingStep(Step):
    DETERMINISTIC = True
    VERSION = "003"
    FORMAT = TorchFormat()

    # TODO: distributed training
    # TODO: recovery of failed jobs (this should be done but needs verification)

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
        model: Lazy[Model],
        dataset: AllenNlpDataset,
        data_loader: Lazy[TangoDataLoader],
        optimizer: Lazy[Optimizer],
        validation_data_loader: Optional[Lazy[TangoDataLoader]] = None,
        training_split: str = "train",
        validation_split: Optional[str] = None,
        patience: Optional[int] = None,
        validation_metric: Union[str, List[str]] = "-loss",
        num_epochs: int = 20,
        checkpointer: Optional[Lazy[Checkpointer]] = None,
        grad_norm: Optional[float] = None,
        grad_clipping: Optional[float] = None,
        learning_rate_scheduler: Optional[Lazy[LearningRateScheduler]] = None,
        momentum_scheduler: Optional[Lazy[MomentumScheduler]] = None,
        moving_average: Optional[Lazy[MovingAverage]] = None,
        callbacks: List[Lazy[TrainerCallback]] = None,
        num_gradient_accumulation_steps: int = 1,
        use_amp: bool = False,
        enable_default_callbacks: bool = True,
        run_sanity_checks: bool = True,
        no_grad: Optional[List[str]] = None,
        limit_batches_per_epoch: Optional[int] = None,
    ) -> Model:
        serialization_dir = self.temp_dir()

        if validation_data_loader is None:
            validation_data_loader = data_loader
        if validation_split is None:
            validation_loader = None
        else:
            validation_data_loader = validation_data_loader.construct(
                instances=dataset.splits[validation_split]
            )
            if limit_batches_per_epoch is not None:
                validation_data_loader = MaxBatchesDataLoader(
                    validation_data_loader, limit_batches_per_epoch
                )
            validation_loader = DataLoaderAdapter(validation_data_loader)

        data_loader = data_loader.construct(instances=dataset.splits[training_split])
        if limit_batches_per_epoch is not None:
            data_loader = MaxBatchesDataLoader(data_loader, limit_batches_per_epoch)
        loader = DataLoaderAdapter(data_loader)

        if cuda.device_count() > 0:
            cuda_device = torch.device(0)
        else:
            cuda_device = torch.device("cpu")
        check_for_gpu(cuda_device)
        loader.set_target_device(cuda_device)
        if validation_loader is not None:
            validation_loader.set_target_device(cuda_device)

        model = model.construct(vocab=dataset.vocab).to(cuda_device)
        if no_grad:
            for name, parameter in model.named_parameters():
                if any(re.search(regex, name) for regex in no_grad):
                    parameter.requires_grad_(False)
        parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        optimizer = optimizer.construct(model_parameters=parameters)
        log_frozen_and_tunable_parameter_names(model)
        moving_average = (
            None if moving_average is None else moving_average.construct(parameters=parameters)
        )
        learning_rate_scheduler = (
            None
            if learning_rate_scheduler is None
            else learning_rate_scheduler.construct(
                optimizer=optimizer,
                num_epochs=num_epochs,
                num_steps_per_epoch=data_loader.num_batches_per_epoch(),
            )
        )
        momentum_scheduler = (
            None
            if momentum_scheduler is None
            else momentum_scheduler.construct(optimizer=optimizer)
        )
        if checkpointer is not None:
            checkpointer = checkpointer.construct(serialization_dir=serialization_dir)
        else:
            checkpointer = Checkpointer(serialization_dir)
        callbacks: List[TrainerCallback] = [
            cb.construct(serialization_dir=serialization_dir) for cb in callbacks or []
        ]

        trainer = GradientDescentTrainer(
            model,
            optimizer=optimizer,
            data_loader=loader,
            patience=patience,
            validation_metric=validation_metric,
            validation_data_loader=validation_loader,
            num_epochs=num_epochs,
            serialization_dir=serialization_dir,
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

        return trainer.model


@Step.register("evaluation")
class EvaluationStep(Step):
    DETERMINISTIC = True
    VERSION = "001"

    @dataclasses.dataclass
    class EvaluationResult:
        metrics: Dict[str, Any]
        predictions: List[Dict[str, Any]]   # TODO: This does not make sense as a type. Should be a List with one element per instance?

    def run(
        self,
        model: Model,
        dataset: AllenNlpDataset,
        split: Optional[str] = "validation",
        data_loader: Optional[Lazy[TangoDataLoader]] = None
    ):
        if data_loader is None:
            data_loader = BatchSizeDataLoader(dataset.splits[split], 32, shuffle=False)
        else:
            data_loader = data_loader.construct(instances=dataset.splits[split])

        if cuda.device_count() > 0:
            cuda_device = torch.device(0)
        else:
            cuda_device = torch.device("cpu")
        check_for_gpu(cuda_device)

        generator_tqdm = Tqdm.tqdm(iter(data_loader))

        # Number of batches in instances.
        batch_results = []
        # Number of batches where the model produces a loss.
        loss_count = 0
        # Cumulative loss
        total_loss = 0.0

        with torch.no_grad():
            model.eval()

            for batch in data_loader:
                batch = move_to_device(batch, cuda_device)
                output_dict = model(**batch)
                batch_results.append(sanitize(output_dict))

                metrics = model.get_metrics()

                loss = output_dict.get("loss")
                if loss is not None:
                    loss_count += 1
                    total_loss += loss.item()
                    metrics["loss"] = total_loss / loss_count

                    if any(metric_name.startswith("_") for metric_name in metrics):
                        logger.warning_once(
                            'Metrics with names beginning with "_" will '
                            "not be logged to the tqdm progress bar."
                        )

                    description = (
                        ", ".join(
                            [
                                "%s: %.2f" % (name, value)
                                for name, value in metrics.items()
                                if not name.startswith("_")
                            ]
                        )
                        + " ||"
                    )
                    generator_tqdm.set_description(description, refresh=False)

        final_metrics = model.get_metrics(reset=True)
        if loss_count > 0:
            # Sanity check
            if loss_count != len(batch_results):
                raise RuntimeError(
                    "The model you are trying to evaluate only sometimes produced a loss!"
                )
            final_metrics["loss"] = total_loss / loss_count

        return self.EvaluationResult(final_metrics, output_dict)
