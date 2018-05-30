# pylint: disable=invalid-name,no-self-use
import os
import sys
from io import StringIO

from allennlp.commands import main
from allennlp.common.testing import AllenNlpTestCase


class TestConfigure(AllenNlpTestCase):

    def test_other_modules(self):
        # Create a new package in a temporary dir
        packagedir = self.TEST_DIR / 'configuretestpackage'
        packagedir.mkdir()  # pylint: disable=no-member
        (packagedir / '__init__.py').touch()  # pylint: disable=no-member

        # And add that directory to the path
        sys.path.insert(0, str(self.TEST_DIR))

        # Write out a duplicate predictor there, but registered under a different name.
        from allennlp.predictors import bidaf
        with open(bidaf.__file__) as f:
            code = f.read().replace("""@Predictor.register('machine-comprehension')""",
                                    """@Predictor.register('configure-test-predictor')""")

        with open(os.path.join(packagedir, 'predictor.py'), 'w') as f:
            f.write(code)

        # Capture stdout
        stdout_saved = sys.stdout
        stdout_captured = StringIO()
        sys.stdout = stdout_captured

        sys.argv = ["run.py",      # executable
                    "configure",     # command
                    "configuretestpackage.predictor.BidafPredictor"]

        main()
        output = stdout_captured.getvalue()
        assert "configure-test-predictor" in output

        sys.stdout = stdout_saved

        sys.path.remove(str(self.TEST_DIR))
