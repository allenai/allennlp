import importlib
import pkgutil
import re
import subprocess

import pytest

import allennlp.commands
from allennlp.common.testing import AllenNlpTestCase


@pytest.mark.skip(reason="This test is slow and somewhat fragile and doesn't need to run every commit.")
class TestDocstringHelp(AllenNlpTestCase):
    RE_DOCSTRING_CALL_SUBCOMMAND_HELP = re.compile(r'^\s*\$ (allennlp \S+ --help)$', re.MULTILINE)
    RE_STARTS_WITH_INDENTATION = re.compile(r'^ {4}', re.MULTILINE)

    def test_docstring_help(self):
        parent_module = allennlp.commands
        for module_info in pkgutil.iter_modules(parent_module.__path__, parent_module.__name__ + '.'):
            module = importlib.import_module(module_info.name)
            match = self.RE_DOCSTRING_CALL_SUBCOMMAND_HELP.search(module.__doc__)
            if match:
                expected_output = self.RE_STARTS_WITH_INDENTATION.sub('', module.__doc__[match.end(0) + 1:])

                str_call_subcommand_help = match.group(1)
                str_call_from_python_subcommand_help = str_call_subcommand_help.replace('allennlp',
                                                                                        'python -m allennlp.run')

                actual_output = subprocess.run(str_call_from_python_subcommand_help, check=True, shell=True,
                                               stdout=subprocess.PIPE).stdout.decode()

                pytorch_pretrained_bert_warning = "Better speed can be achieved with apex installed from" \
                    " https://www.github.com/nvidia/apex.\n"
                if actual_output.startswith(pytorch_pretrained_bert_warning):
                    actual_output = actual_output[len(pytorch_pretrained_bert_warning):]

                self.assertEqual(expected_output, actual_output, f"The documentation for the subcommand usage"
                                                                 f" in the module {module_info.name}"
                                                                 f" does not match the output of running"
                                                                 f" `{str_call_subcommand_help}`."
                                                                 f" Please update the docstring to match the"
                                                                 f" output.")
            else:
                self.assertIn(module_info.name, [parent_module.__name__ + '.subcommand'],
                              f"The documentation for the subcommand usage was not found within the docstring of"
                              f" the module {module_info.name}")
