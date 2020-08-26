"""
Helper functions for archiving models and restoring archived models.
"""
from os import PathLike
from typing import NamedTuple, Union
import atexit
import logging
import os
import tempfile
import tarfile
import shutil

from torch.nn import Module

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.params import Params
from allennlp.models.model import Model, _DEFAULT_WEIGHTS

logger = logging.getLogger(__name__)


class Archive(NamedTuple):
    """ An archive comprises a Model and its experimental config"""

    model: Model
    config: Params

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


def archive_model(
    serialization_dir: Union[str, PathLike],
    weights: str = _DEFAULT_WEIGHTS,
    archive_path: Union[str, PathLike] = None,
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


def load_archive(
    archive_file: str,
    cuda_device: int = -1,
    overrides: str = "",
    weights_file: str = None,
) -> Archive:
    """
    Instantiates an Archive from an archived `tar.gz` file.

    # Parameters

    archive_file : `str`
        The archive file to load the model from.
    cuda_device : `int`, optional (default = `-1`)
        If `cuda_device` is >= 0, the model will be loaded onto the
        corresponding GPU. Otherwise it will be loaded onto the CPU.
    overrides : `str`, optional (default = `""`)
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

    if os.path.isdir(resolved_archive_file):
        serialization_dir = resolved_archive_file
    else:
        # Extract archive to temp dir
        tempdir = tempfile.mkdtemp()
        logger.info(f"extracting archive file {resolved_archive_file} to temp dir {tempdir}")
        with tarfile.open(resolved_archive_file, "r:gz") as archive:
            archive.extractall(tempdir)
        # Postpone cleanup until exit in case the unarchived contents are needed outside
        # this function.
        atexit.register(_cleanup_archive_dir, tempdir)

        serialization_dir = tempdir

    # Load config
    config = Params.from_file(os.path.join(serialization_dir, CONFIG_NAME), overrides)

    if weights_file:
        weights_path = weights_file
    else:
        weights_path = os.path.join(serialization_dir, _WEIGHTS_NAME)
        # Fallback for serialization directories.
        if not os.path.exists(weights_path):
            weights_path = os.path.join(serialization_dir, _DEFAULT_WEIGHTS)

    # Instantiate model. Use a duplicate of the config, as it will get consumed.
    model = Model.load(
        config.duplicate(),
        weights_file=weights_path,
        serialization_dir=serialization_dir,
        cuda_device=cuda_device,
    )

    return Archive(model=model, config=config)


def _cleanup_archive_dir(path: str):
    if os.path.exists(path):
        logger.info("removing temporary unarchived model dir at %s", path)
        shutil.rmtree(path)
