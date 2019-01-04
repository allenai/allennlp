# pylint: disable=invalid-name,no-self-use,protected-access
import argparse
import os
import json
import sys

from allennlp.commands import main, configure as configure_command
from allennlp.common.testing import AllenNlpTestCase
from allennlp.service.config_explorer import make_app


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

        app = None

        # Monkeypatch the run function
        def run_wizard(args: argparse.Namespace) -> None:
            nonlocal app

            app = make_app(args.include_package)
            app.testing = True

        configure_command._run_wizard = run_wizard

        sys.argv = ["run.py",      # executable
                    "configure",     # command
                    "--include-package", "configuretestpackage.predictor"]

        main()

        client = app.test_client()

        response = client.get('/api/config/?class=allennlp.predictors.predictor.Predictor&get_choices=true')
        data = json.loads(response.get_data())
        choices = data.get('choices', ())
        assert 'configuretestpackage.predictor.BidafPredictor' in choices
