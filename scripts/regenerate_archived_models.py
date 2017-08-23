#!/usr/bin/env python

import tarfile
import os
import sys
import logging

from allennlp.models.archival import _CONFIG_NAME, _WEIGHTS_NAME

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def generate_archive(config_file: str,
                     serialization_prefix: str,
                     weights_file: str = 'best.th',
                     archive_name: str = 'model.tar.gz',
                     exist_ok: bool = False) -> None:
    archive_file = os.path.join(serialization_prefix, archive_name)

    if os.path.exists(archive_file):
        if exist_ok:
            logger.info("removing archive file %s", archive_file)
        else:
            logger.error("archive file %s already exists", archive_file)
            sys.exit(-1)

    logger.info("creating new archive file %s", archive_file)

    with tarfile.open(archive_file, 'w:gz') as archive:
        archive.add(config_file, arcname=_CONFIG_NAME)
        archive.add(os.path.join(serialization_prefix, weights_file), arcname=_WEIGHTS_NAME)
        archive.add(os.path.join(serialization_prefix, "vocabulary"), arcname="vocabulary")

if __name__ == "__main__":
    generate_archive("tests/fixtures/bidaf/experiment.json",
                     "tests/fixtures/bidaf/serialization",
                     exist_ok=True)

    generate_archive("tests/fixtures/decomposable_attention/experiment.json",
                     "tests/fixtures/decomposable_attention/serialization",
                     exist_ok=True)

    # GPU model
    generate_archive("tests/fixtures/srl/experiment.json",
                     "tests/fixtures/srl/serialization",
                     exist_ok=True)

    # CPU model
    generate_archive("tests/fixtures/srl/experiment.json",
                     "tests/fixtures/srl/serialization",
                     weights_file="best_cpu.th",
                     archive_name="model_cpu.tar.gz",
                     exist_ok = True)
