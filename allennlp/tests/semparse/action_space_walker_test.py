# pylint: disable=invalid-name
from typing import Set

from allennlp.common.testing import AllenNlpTestCase
from allennlp.semparse import DomainLanguage, ActionSpaceWalker, predicate


class Object:
    pass


class FakeLanguageWithAssertions(DomainLanguage):
    # pylint: disable=unused-argument,no-self-use
    @predicate
    def object_exists(self, items: Set[Object]) -> bool:
        return True

    @predicate
    def black(self, items: Set[Object]) -> Set[Object]:
        return items

    @predicate
    def triangle(self, items: Set[Object]) -> Set[Object]:
        return items

    @predicate
    def touch_wall(self, items: Set[Object]) -> Set[Object]:
        return items

    @predicate
    def all_objects(self) -> Set[Object]:
        return set()


class ActionSpaceWalkerTest(AllenNlpTestCase):
    def setUp(self):
        super(ActionSpaceWalkerTest, self).setUp()
        self.world = FakeLanguageWithAssertions(start_types={bool})
        self.walker = ActionSpaceWalker(self.world, max_path_length=10)

    def test_get_logical_forms_with_agenda(self):
        black_logical_forms = self.walker.get_logical_forms_with_agenda(['<Set[Object]:Set[Object]> -> black'])
        # These are all the possible logical forms with black
        assert len(black_logical_forms) == 25
        shortest_logical_form = self.walker.get_logical_forms_with_agenda(['<Set[Object]:Set[Object]> -> black'],
                                                                          1)[0]
        # This is the shortest complete logical form with black
        assert shortest_logical_form == '(object_exists (black all_objects))'
        agenda = ['<Set[Object]:Set[Object]> -> black',
                  '<Set[Object]:Set[Object]> -> triangle',
                  '<Set[Object]:Set[Object]> -> touch_wall']
        black_triangle_touch_forms = self.walker.get_logical_forms_with_agenda(agenda)
        # Permutations of the three functions. There will not be repetitions of any functions
        # because we limit the length of paths to 10 above.
        assert set(black_triangle_touch_forms) == set([
                '(object_exists (black (triangle (touch_wall all_objects))))',
                '(object_exists (black (touch_wall (triangle all_objects))))',
                '(object_exists (triangle (black (touch_wall all_objects))))',
                '(object_exists (triangle (touch_wall (black all_objects))))',
                '(object_exists (touch_wall (black (triangle all_objects))))',
                '(object_exists (touch_wall (triangle (black all_objects))))'])

    def test_get_logical_forms_with_agenda_and_partial_match(self):
        black_logical_forms = self.walker.get_logical_forms_with_agenda(['<Set[Object]:Set[Object]> -> black'])
        # These are all the possible logical forms with black
        assert len(black_logical_forms) == 25
        shortest_logical_form = self.walker.get_logical_forms_with_agenda(['<Set[Object]:Set[Object]> -> black'],
                                                                          1)[0]
        # This is the shortest complete logical form with black
        assert shortest_logical_form == '(object_exists (black all_objects))'
        agenda = ['<Set[Object]:Set[Object]> -> black',
                  '<Set[Object]:Set[Object]> -> triangle',
                  '<Set[Object]:Set[Object]> -> touch_wall']
        black_triangle_touch_forms = self.walker.get_logical_forms_with_agenda(agenda,
                                                                               allow_partial_match=True)
        # The first six logical forms will contain permutations of all three functions.
        assert set(black_triangle_touch_forms[:6]) == set([
                '(object_exists (black (triangle (touch_wall all_objects))))',
                '(object_exists (black (touch_wall (triangle all_objects))))',
                '(object_exists (triangle (black (touch_wall all_objects))))',
                '(object_exists (triangle (touch_wall (black all_objects))))',
                '(object_exists (touch_wall (black (triangle all_objects))))',
                '(object_exists (touch_wall (triangle (black all_objects))))'])

        # The next six will be the shortest six with two agenda items.
        assert set(black_triangle_touch_forms[6:12]) == set([
                '(object_exists (black (triangle all_objects)))',
                '(object_exists (black (touch_wall all_objects)))',
                '(object_exists (triangle (black all_objects)))',
                '(object_exists (triangle (touch_wall all_objects)))',
                '(object_exists (touch_wall (black all_objects)))',
                '(object_exists (touch_wall (triangle all_objects)))'])

        # After a bunch of longer logical forms with two agenda items, we have the shortest three
        # with one agenda item.
        assert set(black_triangle_touch_forms[30:33]) == set([
                '(object_exists (black all_objects))',
                '(object_exists (triangle all_objects))',
                '(object_exists (touch_wall all_objects))'])

    def test_get_logical_forms_with_empty_agenda_returns_all_logical_forms(self):
        with self.assertLogs("allennlp.semparse.action_space_walker") as log:
            empty_agenda_logical_forms = self.walker.get_logical_forms_with_agenda([],
                                                                                   allow_partial_match=True)
            first_four_logical_forms = empty_agenda_logical_forms[:4]
            assert set(first_four_logical_forms) == {'(object_exists all_objects)',
                                                     '(object_exists (black all_objects))',
                                                     '(object_exists (touch_wall all_objects))',
                                                     '(object_exists (triangle all_objects))'}
        self.assertEqual(log.output,
                         ["WARNING:allennlp.semparse.action_space_walker:"
                          "Agenda is empty! Returning all paths instead."])

    def test_get_logical_forms_with_unmatched_agenda_returns_all_logical_forms(self):
        agenda = ['<Set[Object]:Set[Object]> -> purple']
        with self.assertLogs("allennlp.semparse.action_space_walker") as log:
            empty_agenda_logical_forms = self.walker.get_logical_forms_with_agenda(agenda,
                                                                                   allow_partial_match=True)
            first_four_logical_forms = empty_agenda_logical_forms[:4]
            assert set(first_four_logical_forms) == {'(object_exists all_objects)',
                                                     '(object_exists (black all_objects))',
                                                     '(object_exists (touch_wall all_objects))',
                                                     '(object_exists (triangle all_objects))'}
        self.assertEqual(log.output,
                         ["WARNING:allennlp.semparse.action_space_walker:"
                          "Agenda items not in any of the paths found. Returning all paths."])
        empty_set = self.walker.get_logical_forms_with_agenda(agenda,
                                                              allow_partial_match=False)
        assert empty_set == []

    def test_get_logical_forms_with_agenda_ignores_null_set_item(self):
        with self.assertLogs("allennlp.semparse.action_space_walker") as log:
            agenda = ['<Set[Object]:Set[Object]> -> yellow',
                      '<Set[Object]:Set[Object]> -> black',
                      '<Set[Object]:Set[Object]> -> triangle',
                      '<Set[Object]:Set[Object]> -> touch_wall']
            yellow_black_triangle_touch_forms = self.walker.get_logical_forms_with_agenda(agenda)
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
                          "<Set[Object]:Set[Object]> -> yellow is not in any of the paths found! Ignoring it."])

    def test_get_all_logical_forms(self):
        # get_all_logical_forms should sort logical forms by length.
        ten_shortest_logical_forms = self.walker.get_all_logical_forms(max_num_logical_forms=10)
        shortest_logical_form = ten_shortest_logical_forms[0]
        assert shortest_logical_form == '(object_exists all_objects)'
        length_three_logical_forms = ten_shortest_logical_forms[1:4]
        assert set(length_three_logical_forms) == {'(object_exists (black all_objects))',
                                                   '(object_exists (touch_wall all_objects))',
                                                   '(object_exists (triangle all_objects))'}
