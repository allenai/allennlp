from allennlp.common.testing import AllenNlpTestCase

from allennlp.semparse.domain_languages import QuaRelLanguage
from allennlp.tests.semparse.domain_languages.domain_language_test import check_productions_match


class QuaRelLanguageTest(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.language = QuaRelLanguage()

    def test_infer_quarel(self):
        assert self.language.execute(('(infer (speed higher world1) (friction higher world1) '
                                      '(friction lower world1))')) == 1

        assert self.language.execute(('(infer (speed higher world2) (friction higher world1) '
                                      '(friction lower world1))')) == 0

        # Both answer options are correct.
        assert self.language.execute(('(infer (speed higher world2) (friction higher world1) '
                                      '(friction higher world1))')) == -2

        # Neither answer option is correct.
        assert self.language.execute(('(infer (speed higher world2) (friction higher world2) '
                                      '(friction higher world2))')) == -1

        assert self.language.logical_form_to_action_sequence(('(infer (speed higher world1) '
                                                              '(friction higher world1) '
                                                              '(friction lower world1))')) == \
                ['@start@ -> int',
                 'int -> [<QuaRelType,QuaRelType,QuaRelType:int>, QuaRelType, QuaRelType, QuaRelType]',
                 '<QuaRelType,QuaRelType,QuaRelType:int> -> infer',
                 'QuaRelType -> [<Direction,World:QuaRelType>, Direction, World]',
                 '<Direction,World:QuaRelType> -> speed',
                 'Direction -> higher',
                 'World -> world1',
                 'QuaRelType -> [<Direction,World:QuaRelType>, Direction, World]',
                 '<Direction,World:QuaRelType> -> friction',
                 'Direction -> higher',
                 'World -> world1',
                 'QuaRelType -> [<Direction,World:QuaRelType>, Direction, World]',
                 '<Direction,World:QuaRelType> -> friction',
                 'Direction -> lower',
                 'World -> world1']

    def test_infer_quaval(self):
        assert self.language.execute(('(infer (and (thickness low world1) '
                                      '(thickness high world2)) '
                                      '(strength lower world1) '
                                      '(strength lower world2))')) == 0

        assert self.language.execute(('(infer (and (thickness low world1) '
                                      '(thickness high world2)) '
                                      '(strength lower world2) '
                                      '(strength lower world1))')) == 1

        assert self.language.logical_form_to_action_sequence(('(infer (and (thickness low world1) '
                                                              '(thickness high world2)) '
                                                              '(strength lower world1) '
                                                              '(strength lower world2))')) == \
                ['@start@ -> int',
                 'int -> [<QuaRelType,QuaRelType,QuaRelType:int>, QuaRelType, QuaRelType, '
                 'QuaRelType]',
                 '<QuaRelType,QuaRelType,QuaRelType:int> -> infer',
                 'QuaRelType -> [<QuaRelType,QuaRelType:QuaRelType>, QuaRelType, QuaRelType]',
                 '<QuaRelType,QuaRelType:QuaRelType> -> and',
                 'QuaRelType -> [<Direction,World:QuaRelType>, Direction, World]',
                 '<Direction,World:QuaRelType> -> thickness',
                 'Direction -> low',
                 'World -> world1',
                 'QuaRelType -> [<Direction,World:QuaRelType>, Direction, World]',
                 '<Direction,World:QuaRelType> -> thickness',
                 'Direction -> high',
                 'World -> world2',
                 'QuaRelType -> [<Direction,World:QuaRelType>, Direction, World]',
                 '<Direction,World:QuaRelType> -> strength',
                 'Direction -> lower',
                 'World -> world1',
                 'QuaRelType -> [<Direction,World:QuaRelType>, Direction, World]',
                 '<Direction,World:QuaRelType> -> strength',
                 'Direction -> lower',
                 'World -> world2']

    def test_action_sequence_to_logical_form(self):
        logical_form = ('(infer (and (thickness low world1) '
                        '(thickness high world2)) '
                        '(strength lower world1) '
                        '(strength lower world2))')
        action_sequence = self.language.logical_form_to_action_sequence(logical_form)
        recovered_logical_form = self.language.action_sequence_to_logical_form(action_sequence)
        assert recovered_logical_form == logical_form

    def test_and_incompatible_setup(self):
        logical_form = ('(infer (and (thickness lower world1) '
                        '(thickness lower world2)) '
                        '(strength lower world1) '
                        '(strength lower world2))')
        assert self.language.execute(logical_form) == -1

    def test_get_nonterminal_productions(self):
        productions = self.language.get_nonterminal_productions()
        assert set(productions.keys()) == {
                'QuaRelType',
                '@start@',
                'World',
                'Direction',
                '<QuaRelType,QuaRelType:QuaRelType>',
                'int',
                '<QuaRelType,QuaRelType,QuaRelType:int>',
                '<Direction,World:QuaRelType>'}
        check_productions_match(productions['QuaRelType'],
                                ['[<QuaRelType,QuaRelType:QuaRelType>, QuaRelType, QuaRelType]',
                                 '[<Direction,World:QuaRelType>, Direction, World]'])
        check_productions_match(productions['@start@'],
                                ['int'])
        check_productions_match(productions['World'],
                                ['world1', 'world2'])
        check_productions_match(productions['Direction'],
                                ['higher', 'lower',
                                 'high', 'low'])
        check_productions_match(productions['<QuaRelType,QuaRelType:QuaRelType>'], ['and'])
        check_productions_match(productions['int'],
                                ['[<QuaRelType,QuaRelType,QuaRelType:int>, QuaRelType, QuaRelType, QuaRelType]'])
        check_productions_match(productions['<QuaRelType,QuaRelType,QuaRelType:int>'], ['infer'])
        check_productions_match(productions['<Direction,World:QuaRelType>'],
                                ["friction", "speed", "distance", "heat", "smoothness", "acceleration",
                                 "amountSweat", "apparentSize", "breakability", "brightness", "exerciseIntensity",
                                 "flexibility", "gravity", "loudness", "mass", "strength", "thickness",
                                 "time", "weight"])
