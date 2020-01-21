import logging
import os
import re
from typing import Dict, Iterable, NamedTuple

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import log_frozen_and_tunable_parameter_names, is_master
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
        model: Model = None,
        embedding_sources_mapping: Dict[str, str] = None,
        extend_vocab: bool = False,
    ) -> "TrainerPieces":
        all_datasets = training_util.datasets_from_params(params)

        vocabulary_params = params.pop("vocabulary", {})

        if model:
            if params.pop("model", None):
                logger.warning(
                    "You passed parameters for the model in your configuration file, but we "
                    "are ignoring them, using instead the loaded model parameters."
                )

            # TODO(mattg): This should be updated now that directory_path no longer exists.
            if vocabulary_params.get("directory_path", None):
                logger.warning(
                    "You passed `directory_path` in parameters for the vocabulary in "
                    "your configuration file, but it will be ignored because we already "
                    "have a model with a vocabulary."
                )

            vocab = model.vocab
        else:
            vocab = None

        vocabulary_path = os.path.join(serialization_dir, "vocabulary")

        if not vocab or extend_vocab:
            vocab = TrainerPieces.create_or_extend_vocab(
                datasets=all_datasets,
                params=params,
                recover=recover,
                vocab=vocab,
                vocabulary_params=vocabulary_params,
                vocabulary_path=vocabulary_path,
            )

            if not model:
                model = Model.from_params(vocab=vocab, params=params.pop("model"))

            # If vocab extension is ON for training, embedding extension should also be
            # done. If vocab and embeddings are already in sync, it would be a no-op.
            model.extend_embedder_vocab(embedding_sources_mapping)

        # Initializing the model can have side effect of expanding the vocabulary
        # Save the vocab only in the master. In the degenerate non-distributed
        # case, we're trivially the master.
        if is_master():
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
            model=model,
            iterator=iterator,
            train_dataset=train_data,
            validation_dataset=validation_data,
            test_dataset=test_data,
            validation_iterator=validation_iterator,
            params=trainer_params,
        )

    @staticmethod
    def create_or_extend_vocab(
        params: Params,
        datasets: Dict[str, Iterable[Instance]],
        vocabulary_params: Params,
        vocabulary_path: str,
        vocab: Vocabulary = None,
        recover: bool = False,
    ) -> Vocabulary:
        datasets_for_vocab_creation = set(params.pop("datasets_for_vocab_creation", datasets))

        for key in datasets_for_vocab_creation:
            if key not in datasets:
                raise ConfigurationError(f"invalid 'dataset_for_vocab_creation' {key}")

        instance_generator = (
            instance
            for key, dataset in datasets.items()
            if key in datasets_for_vocab_creation
            for instance in dataset
        )

        dataset_keys_to_use_str = ", ".join(datasets_for_vocab_creation)

        if vocab:
            logger.info(f"Extending model vocabulary using {dataset_keys_to_use_str} data.")
            vocab.extend_from_instances(instances=instance_generator)
        else:
            logger.info(
                "From dataset instances, %s will be considered for vocabulary creation.",
                dataset_keys_to_use_str,
            )

            if recover and os.path.exists(vocabulary_path):
                vocab = Vocabulary.from_files(
                    vocabulary_path,
                    vocabulary_params.get("padding_token", None),
                    vocabulary_params.get("oov_token", None),
                )
            else:
                # Using a generator comprehension here is important because, by being lazy,
                # it allows us to not iterate over the dataset when directory_path is specified.
                vocab = Vocabulary.from_params(vocabulary_params, instances=instance_generator)

        return vocab
