"""
Helper functions for archiving models and restoring archived models.
"""

from typing import NamedTuple
import logging
import os
import tempfile
import tarfile
import shutil

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.models.model import Model, _DEFAULT_WEIGHTS

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# An archive comprises a Model and its experimental config
Archive = NamedTuple("Archive", [("model", Model), ("config", Params)])

# We archive a model by creating a tar.gz file with its weights, config, and vocabulary.
# These are the *known names* under which we archive the config and weights.
_CONFIG_NAME = "config.json"
_WEIGHTS_NAME = "weights.th"

def archive_model(serialization_dir: str,
                  weights: str = _DEFAULT_WEIGHTS) -> None:
    """
    Archives the model weights, its training configuration, and its
    vocabulary to `model.tar.gz`

    Parameters
    ----------
    serialization_dir: ``str``
        The directory where the weights and vocabulary are written out.
    weights: ``str``, optional (default=_DEFAULT_WEIGHTS)
        Which weights file to include in the archive. The default is ``best.th``.
    """
    weights_file = os.path.join(serialization_dir, weights)
    if not os.path.exists(weights_file):
        logger.error("weights file %s does not exist, unable to archive model", weights_file)
        return

    config_file = os.path.join(serialization_dir, "model_params.json")
    if not os.path.exists(config_file):
        logger.error("config file %s does not exist, unable to archive model", config_file)

    archive_file = os.path.join(serialization_dir, "model.tar.gz")
    logger.info("archiving weights and vocabulary to %s", archive_file)
    with tarfile.open(archive_file, 'w:gz') as archive:
        archive.add(config_file, arcname=_CONFIG_NAME)
        archive.add(weights_file, arcname=_WEIGHTS_NAME)
        archive.add(os.path.join(serialization_dir, "vocabulary"),
                    arcname="vocabulary")

def load_archive(archive_file: str, cuda_device: int = -1, overrides: str = "") -> Archive:
    """
    Instantiates an Archive from an archived `tar.gz` file.

    Parameters
    ----------
    archive_file: ``str``
        The archive file to load the model from.
    cuda_device: ``int``, optional (default = -1)
        If `cuda_device` is >= 0, the model will be loaded onto the
        corresponding GPU. Otherwise it will be loaded onto the CPU.
    """
    # redirect to the cache, if necessary
    archive_file = cached_path(archive_file)

    # Extract archive to temp dir
    tempdir = tempfile.mkdtemp()
    logger.info("extracting archive file %s to temp dir %s", archive_file, tempdir)
    with tarfile.open(archive_file, 'r:gz') as archive:
        archive.extractall(tempdir)

    # Load config
    config = Params.from_file(os.path.join(tempdir, _CONFIG_NAME), overrides)

    # Instantiate model. Use a duplicate of the config, as it will get consumed.
    model = Model.load(config.duplicate(),
                       weights_file=os.path.join(tempdir, _WEIGHTS_NAME),
                       serialization_dir=tempdir,
                       cuda_device=cuda_device)

    # Clean up temp dir
    shutil.rmtree(tempdir)

    return Archive(model=model, config=config)
