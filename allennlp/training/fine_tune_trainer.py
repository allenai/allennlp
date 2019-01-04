import logging
import os
import re

from allennlp.commands.evaluate import evaluate
from allennlp.commands.subcommand import Subcommand
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import prepare_environment, prepare_global_logging, \
                                 get_frozen_and_tunable_parameter_names, parse_cuda_device
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.models import load_archive, archive_model
from allennlp.models.archival import CONFIG_NAME
from allennlp.models.model import Model, _DEFAULT_WEIGHTS
from allennlp.training.trainer import Trainer
from allennlp.training.util import datasets_from_params
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.optimizers import Optimizer

logger = logging.getLogger(__name__)

@Trainer.register("fine_tune")
class FineTuneTrainer(Trainer):
    # Requires custom from_params.
    @classmethod
    def from_params(cls,  # type: ignore
                    model: Model,
                    params: Params,
                    serialization_dir: str,
                    extend_vocab: bool = False) -> 'Trainer':
                    # ,
                    #model: Model,
                    # iterator: DataIterator,
                    # train_data: Iterable[Instance],
                    # validation_data: Optional[Iterable[Instance]],
                    # params: Params,
                    # validation_iterator: DataIterator = None) -> 'Trainer':
        # pylint: disable=arguments-differ
        vocabulary_params = params.pop('vocabulary', {})
        if vocabulary_params.get('directory_path', None):
            logger.warning("You passed `directory_path` in parameters for the vocabulary in "
                           "your configuration file, but it will be ignored. ")

        all_datasets = datasets_from_params(params)
        vocab = model.vocab

        if extend_vocab:
            datasets_for_vocab_creation = set(params.pop("datasets_for_vocab_creation", all_datasets))

            for dataset in datasets_for_vocab_creation:
                if dataset not in all_datasets:
                    raise ConfigurationError(f"invalid 'dataset_for_vocab_creation' {dataset}")

            logger.info("Extending model vocabulary using %s data.", ", ".join(datasets_for_vocab_creation))
            vocab.extend_from_instances(vocabulary_params,
                                        (instance for key, dataset in all_datasets.items()
                                        for instance in dataset
                                        if key in datasets_for_vocab_creation))

        # Initializing the model can have side effect of expanding the vocabulary
        vocab.save_to_files(os.path.join(serialization_dir, "vocabulary"))

        iterator = DataIterator.from_params(params.pop("iterator"))
        iterator.index_with(vocab)
        validation_iterator_params = params.pop("validation_iterator", None)
        if validation_iterator_params:
            validation_iterator = DataIterator.from_params(validation_iterator_params)
            validation_iterator.index_with(vocab)
        else:
            validation_iterator = None

        train_data = all_datasets['train']
        validation_data = all_datasets.get('validation')

        trainer_params = params.pop("trainer")
        no_grad_regexes = trainer_params.pop("no_grad", ())
        for name, parameter in model.named_parameters():
            if any(re.search(regex, name) for regex in no_grad_regexes):
                parameter.requires_grad_(False)

        frozen_parameter_names, tunable_parameter_names = \
                    get_frozen_and_tunable_parameter_names(model)
        logger.info("Following parameters are Frozen  (without gradient):")
        for name in frozen_parameter_names:
            logger.info(name)
        logger.info("Following parameters are Tunable (with gradient):")
        for name in tunable_parameter_names:
            logger.info(name)

        patience = trainer_params.pop_int("patience", None)
        validation_metric = trainer_params.pop("validation_metric", "-loss")
        shuffle = trainer_params.pop_bool("shuffle", True)
        num_epochs = trainer_params.pop_int("num_epochs", 20)
        cuda_device = parse_cuda_device(trainer_params.pop("cuda_device", -1))
        grad_norm = trainer_params.pop_float("grad_norm", None)
        grad_clipping = trainer_params.pop_float("grad_clipping", None)
        lr_scheduler_params = trainer_params.pop("learning_rate_scheduler", None)

        if isinstance(cuda_device, list):
            model_device = cuda_device[0]
        else:
            model_device = cuda_device
        if model_device >= 0:
            # Moving model to GPU here so that the optimizer state gets constructed on
            # the right device.
            model = model.cuda(model_device)

        parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        optimizer = Optimizer.from_params(parameters, trainer_params.pop("optimizer"))

        if lr_scheduler_params:
            scheduler = LearningRateScheduler.from_params(optimizer, lr_scheduler_params)
        else:
            scheduler = None

        num_serialized_models_to_keep = trainer_params.pop_int("num_serialized_models_to_keep", 20)
        keep_serialized_model_every_num_seconds = trainer_params.pop_int(
                "keep_serialized_model_every_num_seconds", None)
        model_save_interval = trainer_params.pop_float("model_save_interval", None)
        summary_interval = trainer_params.pop_int("summary_interval", 100)
        histogram_interval = trainer_params.pop_int("histogram_interval", None)
        should_log_parameter_statistics = trainer_params.pop_bool("should_log_parameter_statistics", True)
        should_log_learning_rate = trainer_params.pop_bool("should_log_learning_rate", False)
        log_batch_size_period = trainer_params.pop_int("log_batch_size_period", None)

        params.assert_empty(cls.__name__)
        return cls(model, optimizer, iterator,
                   train_data, validation_data,
                   patience=patience,
                   validation_metric=validation_metric,
                   validation_iterator=validation_iterator,
                   shuffle=shuffle,
                   num_epochs=num_epochs,
                   serialization_dir=serialization_dir,
                   cuda_device=cuda_device,
                   grad_norm=grad_norm,
                   grad_clipping=grad_clipping,
                   learning_rate_scheduler=scheduler,
                   num_serialized_models_to_keep=num_serialized_models_to_keep,
                   keep_serialized_model_every_num_seconds=keep_serialized_model_every_num_seconds,
                   model_save_interval=model_save_interval,
                   summary_interval=summary_interval,
                   histogram_interval=histogram_interval,
                   should_log_parameter_statistics=should_log_parameter_statistics,
                   should_log_learning_rate=should_log_learning_rate,
                   log_batch_size_period=log_batch_size_period)
