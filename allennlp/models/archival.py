"""
Helper functions for archiving models and restoring archived models.
"""

from typing import NamedTuple, Dict, Any
import json
import logging
import os
import tempfile
import tarfile
import shutil

from allennlp.common.file_utils import cached_path
from allennlp.common.params import Params, unflatten, with_fallback, parse_overrides
from allennlp.models.model import Model, _DEFAULT_WEIGHTS

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# An archive comprises a Model and its experimental config
Archive = NamedTuple("Archive", [("model", Model), ("config", Params)])

# We archive a model by creating a tar.gz file with its weights, config, and vocabulary.
#
# We also may include other arbitrary files in the archive. In this case we store
# the mapping { flattened_path -> filename } in ``files_to_archive.json`` and the files
# themselves under the path ``fta/`` .
#
# These constants are the *known names* under which we archive them.
CONFIG_NAME = "config.json"
_WEIGHTS_NAME = "weights.th"
_FTA_NAME = "files_to_archive.json"

def archive_model(serialization_dir: str,
                  weights: str = _DEFAULT_WEIGHTS,
                  files_to_archive: Dict[str, str] = None) -> None:
    """
    Archive the model weights, its training configuration, and its
    vocabulary to `model.tar.gz`. Include the additional ``files_to_archive``
    if provided.

    Parameters
    ----------
    serialization_dir: ``str``
        The directory where the weights and vocabulary are written out.
    weights: ``str``, optional (default=_DEFAULT_WEIGHTS)
        Which weights file to include in the archive. The default is ``best.th``.
    files_to_archive: ``Dict[str, str]``, optional (default=None)
        A mapping {flattened_key -> filename} of supplementary files to include
        in the archive. That is, if you wanted to include ``params['model']['weights']``
        then you would specify the key as `"model.weights"`.
    """
    weights_file = os.path.join(serialization_dir, weights)
    if not os.path.exists(weights_file):
        logger.error("weights file %s does not exist, unable to archive model", weights_file)
        return

    config_file = os.path.join(serialization_dir, CONFIG_NAME)
    if not os.path.exists(config_file):
        logger.error("config file %s does not exist, unable to archive model", config_file)

    # If there are files we want to archive, write out the mapping
    # so that we can use it during de-archiving.
    if files_to_archive:
        fta_filename = os.path.join(serialization_dir, _FTA_NAME)
        with open(fta_filename, 'w') as fta_file:
            fta_file.write(json.dumps(files_to_archive))


    archive_file = os.path.join(serialization_dir, "model.tar.gz")
    logger.info("archiving weights and vocabulary to %s", archive_file)
    with tarfile.open(archive_file, 'w:gz') as archive:
        archive.add(config_file, arcname=CONFIG_NAME)
        archive.add(weights_file, arcname=_WEIGHTS_NAME)
        archive.add(os.path.join(serialization_dir, "vocabulary"),
                    arcname="vocabulary")

        # If there are supplemental files to archive:
        if files_to_archive:
            # Archive the { flattened_key -> original_filename } mapping.
            archive.add(fta_filename, arcname=_FTA_NAME)
            # And add each requested file to the archive.
            for key, filename in files_to_archive.items():
                archive.add(filename, arcname=f"fta/{key}")

def load_archive(archive_file: str,
                 cuda_device: int = -1,
                 overrides: str = "",
                 weights_file: str = None) -> Archive:
    """
    Instantiates an Archive from an archived `tar.gz` file.

    Parameters
    ----------
    archive_file: ``str``
        The archive file to load the model from.
    weights_file: ``str``, optional (default = None)
        The weights file to use.  If unspecified, weights.th in the archive_file will be used.
    cuda_device: ``int``, optional (default = -1)
        If `cuda_device` is >= 0, the model will be loaded onto the
        corresponding GPU. Otherwise it will be loaded onto the CPU.
    overrides: ``str``, optional (default = "")
        JSON overrides to apply to the unarchived ``Params`` object.
    """
    # redirect to the cache, if necessary
    resolved_archive_file = cached_path(archive_file)

    if resolved_archive_file == archive_file:
        logger.info(f"loading archive file {archive_file}")
    else:
        logger.info(f"loading archive file {archive_file} from cache at {resolved_archive_file}")

    tempdir = None
    if os.path.isdir(resolved_archive_file):
        serialization_dir = resolved_archive_file
    else:
        # Extract archive to temp dir
        tempdir = tempfile.mkdtemp()
        logger.info(f"extracting archive file {resolved_archive_file} to temp dir {tempdir}")
        with tarfile.open(resolved_archive_file, 'r:gz') as archive:
            archive.extractall(tempdir)

        serialization_dir = tempdir

    # Check for supplemental files in archive
    fta_filename = os.path.join(serialization_dir, _FTA_NAME)
    if os.path.exists(fta_filename):
        with open(fta_filename, 'r') as fta_file:
            files_to_archive = json.loads(fta_file.read())

        # Add these replacements to overrides
        replacements_dict: Dict[str, Any] = {}
        for key, _ in files_to_archive.items():
            replacement_filename = os.path.join(serialization_dir, f"fta/{key}")
            replacements_dict[key] = replacement_filename

        overrides_dict = parse_overrides(overrides)
        combined_dict = with_fallback(preferred=unflatten(replacements_dict), fallback=overrides_dict)
        overrides = json.dumps(combined_dict)

    # Load config
    config = Params.from_file(os.path.join(serialization_dir, CONFIG_NAME), overrides)
    config.loading_from_archive = True

    if weights_file:
        weights_path = weights_file
    else:
        weights_path = os.path.join(serialization_dir, _WEIGHTS_NAME)

    # Instantiate model. Use a duplicate of the config, as it will get consumed.
    model = Model.load(config.duplicate(),
                       weights_file=weights_path,
                       serialization_dir=serialization_dir,
                       cuda_device=cuda_device)

    if tempdir:
        # Clean up temp dir
        shutil.rmtree(tempdir)

    return Archive(model=model, config=config)
