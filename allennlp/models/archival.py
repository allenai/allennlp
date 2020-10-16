"""
Helper functions for archiving models and restoring archived models.
"""
from os import PathLike
from typing import NamedTuple, Union, Dict, Any, List, Optional
import logging
import os
import tempfile
import tarfile
import shutil
from pathlib import Path
from contextlib import contextmanager

from torch.nn import Module

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.params import Params
from allennlp.data.dataset_readers import DatasetReader
from allennlp.models.model import Model, _DEFAULT_WEIGHTS

logger = logging.getLogger(__name__)


class Archive(NamedTuple):
    """ An archive comprises a Model and its experimental config"""

    model: Model
    config: Params
    dataset_reader: DatasetReader
    validation_dataset_reader: DatasetReader

    def extract_module(self, path: str, freeze: bool = True) -> Module:
        """
        This method can be used to load a module from the pretrained model archive.

        It is also used implicitly in FromParams based construction. So instead of using standard
        params to construct a module, you can instead load a pretrained module from the model
        archive directly. For eg, instead of using params like {"type": "module_type", ...}, you
        can use the following template::

            {
                "_pretrained": {
                    "archive_file": "../path/to/model.tar.gz",
                    "path": "path.to.module.in.model",
                    "freeze": False
                }
            }

        If you use this feature with FromParams, take care of the following caveat: Call to
        initializer(self) at end of model initializer can potentially wipe the transferred parameters
        by reinitializing them. This can happen if you have setup initializer regex that also
        matches parameters of the transferred module. To safe-guard against this, you can either
        update your initializer regex to prevent conflicting match or add extra initializer::

            [
                [".*transferred_module_name.*", "prevent"]]
            ]

        # Parameters

        path : `str`, required
            Path of target module to be loaded from the model.
            Eg. "_textfield_embedder.token_embedder_tokens"
        freeze : `bool`, optional (default=`True`)
            Whether to freeze the module parameters or not.

        """
        modules_dict = {path: module for path, module in self.model.named_modules()}
        module = modules_dict.get(path)

        if not module:
            raise ConfigurationError(
                f"You asked to transfer module at path {path} from "
                f"the model {type(self.model)}. But it's not present."
            )
        if not isinstance(module, Module):
            raise ConfigurationError(
                f"The transferred object from model {type(self.model)} at path "
                f"{path} is not a PyTorch Module."
            )

        for parameter in module.parameters():  # type: ignore
            parameter.requires_grad_(not freeze)
        return module


# We archive a model by creating a tar.gz file with its weights, config, and vocabulary.
#
# These constants are the *known names* under which we archive them.
CONFIG_NAME = "config.json"
_WEIGHTS_NAME = "weights.th"


def verify_include_in_archive(include_in_archive: Optional[List[str]] = None):
    if include_in_archive is None:
        return
    saved_names = [CONFIG_NAME, _WEIGHTS_NAME, _DEFAULT_WEIGHTS, "vocabulary"]
    for archival_target in include_in_archive:
        if archival_target in saved_names:
            raise ConfigurationError(
                f"{', '.join(saved_names)} are saved names and cannot be used for include_in_archive."
            )


def archive_model(
    serialization_dir: Union[str, PathLike],
    weights: str = _DEFAULT_WEIGHTS,
    archive_path: Union[str, PathLike] = None,
    include_in_archive: Optional[List[str]] = None,
) -> None:
    """
    Archive the model weights, its training configuration, and its vocabulary to `model.tar.gz`.

    # Parameters

    serialization_dir : `str`
        The directory where the weights and vocabulary are written out.
    weights : `str`, optional (default=`_DEFAULT_WEIGHTS`)
        Which weights file to include in the archive. The default is `best.th`.
    archive_path : `str`, optional, (default = `None`)
        A full path to serialize the model to. The default is "model.tar.gz" inside the
        serialization_dir. If you pass a directory here, we'll serialize the model
        to "model.tar.gz" inside the directory.
    """
    weights_file = os.path.join(serialization_dir, weights)
    if not os.path.exists(weights_file):
        logger.error("weights file %s does not exist, unable to archive model", weights_file)
        return

    config_file = os.path.join(serialization_dir, CONFIG_NAME)
    if not os.path.exists(config_file):
        logger.error("config file %s does not exist, unable to archive model", config_file)

    if archive_path is not None:
        archive_file = archive_path
        if os.path.isdir(archive_file):
            archive_file = os.path.join(archive_file, "model.tar.gz")
    else:
        archive_file = os.path.join(serialization_dir, "model.tar.gz")
    logger.info("archiving weights and vocabulary to %s", archive_file)
    with tarfile.open(archive_file, "w:gz") as archive:
        archive.add(config_file, arcname=CONFIG_NAME)
        archive.add(weights_file, arcname=_WEIGHTS_NAME)
        archive.add(os.path.join(serialization_dir, "vocabulary"), arcname="vocabulary")

        if include_in_archive is not None:
            for archival_target in include_in_archive:
                archival_target_path = os.path.join(serialization_dir, archival_target)
                if os.path.exists(archival_target_path):
                    archive.add(archival_target_path, arcname=archival_target)


def load_archive(
    archive_file: Union[str, Path],
    cuda_device: int = -1,
    overrides: Union[str, Dict[str, Any]] = "",
    weights_file: str = None,
) -> Archive:
    """
    Instantiates an Archive from an archived `tar.gz` file.

    # Parameters

    archive_file : `Union[str, Path]`
        The archive file to load the model from.
    cuda_device : `int`, optional (default = `-1`)
        If `cuda_device` is >= 0, the model will be loaded onto the
        corresponding GPU. Otherwise it will be loaded onto the CPU.
    overrides : `Union[str, Dict[str, Any]]`, optional (default = `""`)
        JSON overrides to apply to the unarchived `Params` object.
    weights_file : `str`, optional (default = `None`)
        The weights file to use.  If unspecified, weights.th in the archive_file will be used.
    """
    # redirect to the cache, if necessary
    resolved_archive_file = cached_path(archive_file)

    if resolved_archive_file == archive_file:
        logger.info(f"loading archive file {archive_file}")
    else:
        logger.info(f"loading archive file {archive_file} from cache at {resolved_archive_file}")

    tempdir = None
    try:
        if os.path.isdir(resolved_archive_file):
            serialization_dir = resolved_archive_file
        else:
            with extracted_archive(resolved_archive_file, cleanup=False) as tempdir:
                serialization_dir = tempdir

        if weights_file:
            weights_path = weights_file
        else:
            weights_path = get_weights_path(serialization_dir)

        # Load config
        config = Params.from_file(os.path.join(serialization_dir, CONFIG_NAME), overrides)

        # Instantiate model and dataset readers. Use a duplicate of the config, as it will get consumed.
        dataset_reader, validation_dataset_reader = _load_dataset_readers(config.duplicate(), serialization_dir)
        model = _load_model(config.duplicate(), weights_path, serialization_dir, cuda_device)
    finally:
        if tempdir is not None:
            logger.info(f"removing temporary unarchived model dir at {tempdir}")
            shutil.rmtree(tempdir, ignore_errors=True)

    return Archive(
        model=model,
        config=config,
        dataset_reader=dataset_reader,
        validation_dataset_reader=validation_dataset_reader,
    )


def _load_dataset_readers(config, serialization_dir):
    dataset_reader_params = config.get("dataset_reader")

    # Try to use the validation dataset reader if there is one - otherwise fall back
    # to the default dataset_reader used for both training and validation.
    validation_dataset_reader_params = config.get(
        "validation_dataset_reader", dataset_reader_params.duplicate()
    )

    dataset_reader = DatasetReader.from_params(dataset_reader_params, serialization_dir)
    validation_dataset_reader = DatasetReader.from_params(validation_dataset_reader_params, serialization_dir)

    return dataset_reader, validation_dataset_reader


def _load_model(config, weights_path, serialization_dir, cuda_device):
    return Model.load(
        config,
        weights_file=weights_path,
        serialization_dir=serialization_dir,
        cuda_device=cuda_device,
    )


def get_weights_path(serialization_dir):
    weights_path = os.path.join(serialization_dir, _WEIGHTS_NAME)
    # Fallback for serialization directories.
    if not os.path.exists(weights_path):
        weights_path = os.path.join(serialization_dir, _DEFAULT_WEIGHTS)
    return weights_path


@contextmanager
def extracted_archive(resolved_archive_file, cleanup=True):
    tempdir = None
    try:
        tempdir = tempfile.mkdtemp()
        logger.info(f"extracting archive file {resolved_archive_file} to temp dir {tempdir}")
        with tarfile.open(resolved_archive_file, "r:gz") as archive:
            archive.extractall(tempdir)
        yield tempdir
    finally:
        if tempdir is not None and cleanup:
            logger.info(f"removing temporary unarchived model dir at {tempdir}")
            shutil.rmtree(tempdir, ignore_errors=True)
