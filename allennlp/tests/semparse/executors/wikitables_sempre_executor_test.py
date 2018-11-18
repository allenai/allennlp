# pylint: disable=no-self-use,invalid-name,protected-access
import os
import pytest

from allennlp.common.testing import AllenNlpTestCase
from allennlp.semparse.executors import WikiTablesSempreExecutor
from allennlp.semparse.executors.wikitables_sempre_executor import SEMPRE_ABBREVIATIONS_PATH, SEMPRE_GRAMMAR_PATH

@pytest.mark.java
class WikiTablesSempreExecutorTest(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.should_remove_sempre_abbreviations = not os.path.exists(SEMPRE_ABBREVIATIONS_PATH)
        self.should_remove_sempre_grammar = not os.path.exists(SEMPRE_GRAMMAR_PATH)

    def tearDown(self):
        super().tearDown()
        if self.should_remove_sempre_abbreviations and os.path.exists(SEMPRE_ABBREVIATIONS_PATH):
            os.remove(SEMPRE_ABBREVIATIONS_PATH)
        if self.should_remove_sempre_grammar and os.path.exists(SEMPRE_GRAMMAR_PATH):
            os.remove(SEMPRE_GRAMMAR_PATH)

    def test_accuracy_is_scored_correctly(self):
        # This is the first example in our test fixture.
        example_string = ('(example (id nt-0) (utterance "what was the last year where this team '
                          'was a part of the usl a-league?") (context (graph '
                          'tables.TableKnowledgeGraph tables/590.csv)) '
                          '(targetValue (list (description "2004"))))')

        # This logical form should produce the correct denotation (the "targetValue" above) given
        # the table.
        logical_form = ('((reverse fb:row.row.year) (fb:row.row.index (max '
                        '((reverse fb:row.row.index) (fb:row.row.league fb:cell.usl_a_league)))))')
        executor = WikiTablesSempreExecutor(table_directory=str(self.FIXTURES_ROOT / 'data' / 'wikitables/'))
        assert executor.evaluate_logical_form(logical_form, example_string) is True

        # Testing that we handle bad logical forms correctly.
        assert executor.evaluate_logical_form(None, example_string) is False

        assert executor.evaluate_logical_form('Error producing logical form', example_string) is False

        # And an incorrect logical form.
        logical_form = '(fb:row.row.league fb:cell.3rd_usl_3rd)'
        assert executor.evaluate_logical_form(logical_form, example_string) is False
