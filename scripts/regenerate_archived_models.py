#!/usr/bin/env python

import tarfile
import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from allennlp.models.archival import _CONFIG_NAME, _WEIGHTS_NAME

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def generate_archive(config_file: str,
                     serialization_dir: str,
                     weights_file: str = 'best.th',
                     archive_name: str = 'model.tar.gz',
                     exist_ok: bool = False) -> None:
    archive_file = os.path.join(serialization_dir, archive_name)

    if os.path.exists(archive_file):
        if exist_ok:
            logger.info("removing archive file %s", archive_file)
        else:
            logger.error("archive file %s already exists", archive_file)
            sys.exit(-1)

    logger.info("creating new archive file %s", archive_file)

    with tarfile.open(archive_file, 'w:gz') as archive:
        archive.add(config_file, arcname=_CONFIG_NAME)
        archive.add(os.path.join(serialization_dir, weights_file), arcname=_WEIGHTS_NAME)
        archive.add(os.path.join(serialization_dir, "vocabulary"), arcname="vocabulary")

if __name__ == "__main__":
    generate_archive("tests/fixtures/bidaf/experiment.json",
                     "tests/fixtures/bidaf/serialization",
                     exist_ok=True)

    generate_archive("tests/fixtures/decomposable_attention/experiment.json",
                     "tests/fixtures/decomposable_attention/serialization",
                     exist_ok=True)

    generate_archive("tests/fixtures/srl/experiment.json",
                     "tests/fixtures/srl/serialization",
                     exist_ok=True)
