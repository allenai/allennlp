import logging
import os
import re
from typing import Dict, Iterable, NamedTuple

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import log_frozen_and_tunable_parameter_names
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.training import util as training_util

logger = logging.getLogger(__name__)


class TrainerPieces(NamedTuple):
    """
    We would like to avoid having complex instantiation logic taking place
    in `Trainer.from_params`. This helper class has a `from_params` that
    instantiates a model, loads train (and possibly validation and test) datasets,
    constructs a Vocabulary, creates data iterators, and handles a little bit
    of bookkeeping. If you're creating your own alternative training regime
    you might be able to use this.
    """

    model: Model
    iterator: DataIterator
    train_dataset: Iterable[Instance]
    validation_dataset: Iterable[Instance]
    test_dataset: Iterable[Instance]
    validation_iterator: DataIterator
    params: Params

    @classmethod
    def from_params(
        cls,
        params: Params,
        serialization_dir: str,
        recover: bool = False,
        cache_directory: str = None,
        cache_prefix: str = None,
        model: Model = None,
        embedding_sources_mapping: Dict[str, str] = None,
        extend_vocab: bool = False,
    ) -> "TrainerPieces":
        all_datasets = training_util.datasets_from_params(params, cache_directory, cache_prefix)

        vocabulary_params = params.pop("vocabulary", {})

        if model:
            if params.pop("model", None):
                logger.warning(
                    "You passed parameters for the model in your configuration file, but we "
                    "are ignoring them, using instead the model parameters in the archive."
                )

            if vocabulary_params.get("directory_path", None):
                logger.warning(
                    "You passed `directory_path` in parameters for the vocabulary in "
                    "your configuration file, but it will be ignored. "
                )

            vocab = model.vocab
        else:
            vocab = None

        vocabulary_path = os.path.join(serialization_dir, "vocabulary")

        if not vocab or extend_vocab:
            datasets_for_vocab_creation = set(
                params.pop("datasets_for_vocab_creation", all_datasets)
            )

            for dataset in datasets_for_vocab_creation:
                if dataset not in all_datasets:
                    raise ConfigurationError(f"invalid 'dataset_for_vocab_creation' {dataset}")

            instance_generator = (
                instance
                for key, dataset in all_datasets.items()
                if key in datasets_for_vocab_creation
                for instance in dataset
            )

            if vocab:
                logger.info(
                    f"Extending model vocabulary using {', '.join(datasets_for_vocab_creation)} data."
                )
                vocab.extend_from_instances(vocabulary_params, instance_generator)
            else:
                logger.info(
                    "From dataset instances, %s will be considered for vocabulary creation.",
                    ", ".join(datasets_for_vocab_creation),
                )

                if recover and os.path.exists(vocabulary_path):
                    vocab = Vocabulary.from_files(vocabulary_path)
                else:
                    # Using a generator comprehension here is important  because, being lazy,
                    # it allows us to not iterate over the dataset when directory_path is specified.
                    vocab = Vocabulary.from_params(vocabulary_params, instance_generator)

                assert model is None
                model = Model.from_params(vocab=vocab, params=params.pop("model"))

            # If vocab extension is ON for training, embedding extension should also be
            # done. If vocab and embeddings are already in sync, it would be a no-op.
            model.extend_embedder_vocab(embedding_sources_mapping)

        # Initializing the model can have side effect of expanding the vocabulary
        vocab.save_to_files(vocabulary_path)

        iterator = DataIterator.from_params(params.pop("iterator"))
        iterator.index_with(model.vocab)
        validation_iterator_params = params.pop("validation_iterator", None)
        if validation_iterator_params:
            validation_iterator = DataIterator.from_params(validation_iterator_params)
            validation_iterator.index_with(model.vocab)
        else:
            validation_iterator = None

        train_data = all_datasets["train"]
        validation_data = all_datasets.get("validation")
        test_data = all_datasets.get("test")

        trainer_params = params.pop("trainer")
        no_grad_regexes = trainer_params.pop("no_grad", ())
        for name, parameter in model.named_parameters():
            if any(re.search(regex, name) for regex in no_grad_regexes):
                parameter.requires_grad_(False)

        log_frozen_and_tunable_parameter_names(model)

        return cls(
            model,
            iterator,
            train_data,
            validation_data,
            test_data,
            validation_iterator,
            trainer_params,
        )
