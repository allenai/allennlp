# pylint: disable=invalid-name
from overrides import overrides

from nltk.sem.logic import TRUTH_TYPE

from allennlp.common.testing import AllenNlpTestCase
from allennlp.semparse.type_declarations.type_declaration import NamedBasicType
from allennlp.semparse.worlds.world import World
from allennlp.semparse import ActionSpaceWalker


class FakeWorldWithAssertions(World):
    # pylint: disable=abstract-method
    @overrides
    def get_valid_starting_types(self):
        return set([TRUTH_TYPE])

    @overrides
    def get_basic_types(self):
        return set([NamedBasicType("OBJECT")])

    @overrides
    def get_valid_actions(self):
        # This grammar produces true/false statements like
        # (object_exists all_objects)
        # (object_exists (black all_objects))
        # (object_exists (triangle all_objects))
        # (object_exists (touch_wall all_objects))
        # (object_exists (triangle (black all_objects)))
        # ...
        actions = {'@start@': ['@start@ -> t'],
                   't': ['t -> [<o,t>, o]'],
                   '<o,t>': ['<o,t> -> object_exists'],
                   'o': ['o -> [<o,o>, o]', 'o -> all_objects'],
                   '<o,o>': ['<o,o> -> black', '<o,o> -> triangle', '<o,o> -> touch_wall']}
        return actions

    @overrides
    def is_terminal(self, symbol: str) -> bool:
        return symbol in {'object_exists', 'all_objects', 'black', 'triangle', 'touch_wall'}


class ActionSpaceWalkerTest(AllenNlpTestCase):
    def setUp(self):
        super(ActionSpaceWalkerTest, self).setUp()
        self.world = FakeWorldWithAssertions()
        self.walker = ActionSpaceWalker(self.world, max_path_length=10)

    def test_get_logical_forms_with_agenda(self):
        black_logical_forms = self.walker.get_logical_forms_with_agenda(['<o,o> -> black'])
        # These are all the possible logical forms with black
        assert len(black_logical_forms) == 25
        shortest_logical_form = self.walker.get_logical_forms_with_agenda(['<o,o> -> black'], 1)[0]
        # This is the shortest complete logical form with black
        assert shortest_logical_form == '(object_exists (black all_objects))'
        black_triangle_touch_forms = self.walker.get_logical_forms_with_agenda(['<o,o> -> black',
                                                                                '<o,o> -> triangle',
                                                                                '<o,o> -> touch_wall'])
        # Permutations of the three functions. There will not be repetitions of any functions
        # because we limit the length of paths to 10 above.
        assert set(black_triangle_touch_forms) == set([
                '(object_exists (black (triangle (touch_wall all_objects))))',
                '(object_exists (black (touch_wall (triangle all_objects))))',
                '(object_exists (triangle (black (touch_wall all_objects))))',
                '(object_exists (triangle (touch_wall (black all_objects))))',
                '(object_exists (touch_wall (black (triangle all_objects))))',
                '(object_exists (touch_wall (triangle (black all_objects))))'])

    def test_get_logical_forms_with_empty_agenda_returns_all_logical_forms(self):
        with self.assertLogs("allennlp.semparse.action_space_walker") as log:
            empty_agenda_logical_forms = self.walker.get_logical_forms_with_agenda([])
            first_four_logical_forms = empty_agenda_logical_forms[:4]
            assert set(first_four_logical_forms) == {'(object_exists all_objects)',
                                                     '(object_exists (black all_objects))',
                                                     '(object_exists (touch_wall all_objects))',
                                                     '(object_exists (triangle all_objects))'}
        self.assertEqual(log.output,
                         ["WARNING:allennlp.semparse.action_space_walker:"
                          "Agenda is empty! Returning all paths instead."])

    def test_get_logical_forms_with_agenda_ignores_null_set_item(self):
        with self.assertLogs("allennlp.semparse.action_space_walker") as log:
            yellow_black_triangle_touch_forms = self.walker.get_logical_forms_with_agenda(['<o,o> -> yellow',
                                                                                           '<o,o> -> black',
                                                                                           '<o,o> -> triangle',
                                                                                           '<o,o> -> touch_wall'])
            # Permutations of the three functions, after ignoring yellow. There will not be repetitions
            # of any functions because we limit the length of paths to 10 above.
            assert set(yellow_black_triangle_touch_forms) == set([
                    '(object_exists (black (triangle (touch_wall all_objects))))',
                    '(object_exists (black (touch_wall (triangle all_objects))))',
                    '(object_exists (triangle (black (touch_wall all_objects))))',
                    '(object_exists (triangle (touch_wall (black all_objects))))',
                    '(object_exists (touch_wall (black (triangle all_objects))))',
                    '(object_exists (touch_wall (triangle (black all_objects))))'])
        self.assertEqual(log.output,
                         ["WARNING:allennlp.semparse.action_space_walker:"
                          "<o,o> -> yellow is not in any of the paths found! Ignoring it."])

    def test_get_all_logical_forms(self):
        # get_all_logical_forms should sort logical forms by length.
        ten_shortest_logical_forms = self.walker.get_all_logical_forms(max_num_logical_forms=10)
        shortest_logical_form = ten_shortest_logical_forms[0]
        assert shortest_logical_form == '(object_exists all_objects)'
        length_three_logical_forms = ten_shortest_logical_forms[1:4]
        assert set(length_three_logical_forms) == {'(object_exists (black all_objects))',
                                                   '(object_exists (touch_wall all_objects))',
                                                   '(object_exists (triangle all_objects))'}
