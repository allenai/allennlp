import logging
import os
import re
from typing import Iterable, NamedTuple

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import get_frozen_and_tunable_parameter_names
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.training import util as training_util

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


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
    def from_params(cls,
                    params: Params,
                    serialization_dir: str,
                    recover: bool = False,
                    cache_directory: str = None,
                    cache_prefix: str = None) -> 'TrainerPieces':
        all_datasets = training_util.datasets_from_params(params, cache_directory, cache_prefix)
        datasets_for_vocab_creation = set(params.pop("datasets_for_vocab_creation", all_datasets))

        for dataset in datasets_for_vocab_creation:
            if dataset not in all_datasets:
                raise ConfigurationError(f"invalid 'dataset_for_vocab_creation' {dataset}")

        logger.info("From dataset instances, %s will be considered for vocabulary creation.",
                    ", ".join(datasets_for_vocab_creation))

        if recover and os.path.exists(os.path.join(serialization_dir, "vocabulary")):
            vocab = Vocabulary.from_files(os.path.join(serialization_dir, "vocabulary"))
            params.pop("vocabulary", {})
        else:
            vocab = Vocabulary.from_params(
                    params.pop("vocabulary", {}),
                    # Using a generator comprehension here is important
                    # because, being lazy, it allows us to not iterate over the
                    # dataset when directory_path is specified.
                    (instance for key, dataset in all_datasets.items()
                     if key in datasets_for_vocab_creation for instance in dataset)
            )

        model = Model.from_params(vocab=vocab, params=params.pop('model'))

        # If vocab extension is ON for training, embedding extension should also be
        # done. If vocab and embeddings are already in sync, it would be a no-op.
        model.extend_embedder_vocab()

        # Initializing the model can have side effect of expanding the vocabulary
        vocab.save_to_files(os.path.join(serialization_dir, "vocabulary"))

        iterator = DataIterator.from_params(params.pop("iterator"))
        iterator.index_with(model.vocab)
        validation_iterator_params = params.pop("validation_iterator", None)
        if validation_iterator_params:
            validation_iterator = DataIterator.from_params(validation_iterator_params)
            validation_iterator.index_with(model.vocab)
        else:
            validation_iterator = None

        train_data = all_datasets['train']
        validation_data = all_datasets.get('validation')
        test_data = all_datasets.get('test')

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

        return cls(model, iterator,
                   train_data, validation_data, test_data,
                   validation_iterator, trainer_params)
