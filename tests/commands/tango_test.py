import gzip
import os
import sys
import time

import dill

from allennlp.commands import main
from allennlp.common.testing import AllenNlpTestCase


class TestTangoCommand(AllenNlpTestCase):
    def test_tango(self):
        output_path = self.TEST_DIR / "tango_serialization_dir"
        sys.argv = [
            "allennlp",
            "tango",
            str(self.FIXTURES_ROOT / "tango" / "train_tagger.jsonnet"),
            "-s",
            str(output_path),
        ]

        start = time.time()
        main()
        first_run_time = time.time() - start

        assert os.path.exists(output_path)
        for symlink_name in {"dataset", "evaluation", "trained_model"}:
            assert os.path.islink(output_path / symlink_name)

        with gzip.open(output_path / "evaluation" / "data.dill.gz") as f:
            dill.load(f)
            is_iterator = dill.load(f)
            assert is_iterator is False
            evaluation_result = dill.load(f)
            assert evaluation_result.metrics["accuracy"] == 1.0

        # If we try again, it should run faster, because everything is cached.
        start = time.time()
        main()
        second_run_time = time.time() - start

        assert second_run_time * 2 < first_run_time
