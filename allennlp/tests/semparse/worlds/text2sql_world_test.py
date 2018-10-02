# pylint: disable=too-many-lines,invalid-name

from allennlp.common.testing import AllenNlpTestCase
from allennlp.semparse.worlds.text2sql_world import Text2SqlWorld
from allennlp.semparse.contexts.text2sql_table_context import Text2SqlTableContext


class TestText2SqlWorld(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.schema = str(self.FIXTURES_ROOT / 'data' / 'text2sql' / 'restaurants-schema.csv')
        context = Text2SqlTableContext(self.schema)
        self.world = Text2SqlWorld(context)

    def test_get_action_sequence_and_global_actions(self):
        # TODO(Mark): Fill in this test.
        action_sequence, all_actions = self.world.get_action_sequence_and_all_actions(None)

        assert action_sequence == []
        assert all_actions is not None
