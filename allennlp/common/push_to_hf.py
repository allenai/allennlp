"""
Utilities for pushing models to the Hugging Face Hub ([hf.co](https://hf.co/)).
"""

import logging
import sys
from typing import Optional, Union
from pathlib import Path

from allennlp.common.file_utils import cached_path
import shutil

import zipfile
import tarfile
import tempfile

from huggingface_hub import Repository, HfApi, HfFolder

logger = logging.getLogger(__name__)

README_TEMPLATE = """---
tags:
- allennlp
---

# TODO: Fill this model card
"""


def _create_model_card(repo_dir: Path):
    """Creates a model card for the repository.

    TODO: Add metrics to model-index
    TODO: Use information from common model cards
    """
    readme_path = repo_dir / "README.md"
    prev_readme = ""
    if readme_path.exists():
        with readme_path.open("r", encoding="utf8") as f:
            prev_readme = f.read()
    with readme_path.open("w", encoding="utf-8") as f:
        f.write(README_TEMPLATE)
        f.write(prev_readme)


_ALLOWLIST_PATHS = ["vocabulary", "config.json", "weights.th", "best.th", "metrics.json", "log"]


def _copy_allowed_file(filepath: Path, dst_directory: Path):
    """
    Copies files from allowlist to a directory, overriding existing
    files or directories if any.
    """
    if filepath.name not in _ALLOWLIST_PATHS:
        return

    dst = dst_directory / filepath.name
    if dst.is_dir():
        shutil.rmtree(str(dst))
    elif dst.is_file():
        dst.unlink()
    if filepath.is_dir():
        shutil.copytree(filepath, dst)
    elif filepath.is_file():
        if filepath.name == "best.th":
            dst = dst_directory / "model.th"
        shutil.copy(str(filepath), str(dst))


def push_to_hf(
    archive_path: Union[str, Path],
    repo_name: str,
    organization: Optional[str] = None,
    commit_message: str = "Update repository",
    local_repo_path: Union[str, Path] = "hub",
):
    """Pushes model and related files to the Hugging Face Hub ([hf.co](https://hf.co/))

    # Parameters

    archive_path : `Union[str, Path]`
        Full path to the zipped model (e.g. model/model.tar.gz) or to a directory with the serialized model.

    repo_name: `str`
        Name of the repository in the Hugging Face Hub.

    organization : `Optional[str]`, optional (default = `None`)
        Name of organization to which the model should be uploaded.

    commit_message: `str` (default=`Update repository`)
        Commit message to use for the push.

    local_repo_path : `Union[str, Path]`, optional (default=`hub`)
        Local directory where the repository will be saved.

    """
    archive_path = Path(archive_path)

    if not archive_path.exists():
        logging.error(
            f"Can't find archive path: {archive_path}, please"
            "point to either a .tar.gz archive or to a directory"
            "with the serialized model."
        )
        sys.exit(1)

    # Create the repo (or clone its content if it's nonempty)
    api = HfApi()
    repo_url = api.create_repo(
        name=repo_name,
        token=HfFolder.get_token(),
        organization=organization,
        private=False,
        exist_ok=True,
    )

    repo_local_path = Path(local_repo_path) / repo_name
    repo = Repository(repo_local_path, clone_from=repo_url)
    repo.git_pull(rebase=True)

    # Model file should be tracked with Git LFS
    repo.lfs_track(["*.th"])
    info_msg = f"Preparing repository '{repo_name}'"
    if organization is not None:
        info_msg += f" ({organization})"
    logging.info(info_msg)

    # Extract information from either serializable directory or a
    # .tar.gz file
    if archive_path.is_dir():
        for filename in archive_path.iterdir():
            _copy_allowed_file(Path(filename), repo_local_path)
    elif zipfile.is_zipfile(archive_path) or tarfile.is_tarfile(archive_path):
        with tempfile.TemporaryDirectory() as temp_dir:
            extracted_dir = Path(cached_path(archive_path, temp_dir, extract_archive=True))
            for filename in extracted_dir.iterdir():
                _copy_allowed_file(Path(filename), repo_local_path)

    _create_model_card(repo_local_path)

    logging.info(f"Pushing repo {repo_name} to the Hugging Face Hub")
    url = repo.push_to_hub(commit_message=commit_message)

    url, _ = url.split("/commit/")
    logging.info(f"View your model in {url}")
