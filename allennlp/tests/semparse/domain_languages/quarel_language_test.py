
from allennlp.common.testing import AllenNlpTestCase
from allennlp.semparse.domain_languages.quarel_domain_language import QuaRel

class QuaRelLanguageTest(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.language = QuaRel()

    def test_infer_quarel(self):
        assert self.language.execute(('(infer (speed higher world1) (friction higher world1) '
                                      '(friction lower world1))')) == 1
        assert self.language.logical_form_to_action_sequence(('(infer (speed higher world1) '
                                                              '(friction higher world1) '
                                                              '(friction lower world1))')) == \
                ['@start@ -> int',
                 'int -> [<QuaRelType,QuaRelType,QuaRelType:int>, QuaRelType, QuaRelType, '
                 'QuaRelType]',
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

        assert self.language.execute(('(infer (speed higher world2) (friction higher world1) '
                                      '(friction lower world1))')) == 0

    def test_infer_quaval(self):
        assert self.language.execute(('(infer (and (thickness low world1) '
                                      '(thickness high world2)) '
                                      '(strength lower world1) '
                                      '(strength lower world2))')) == 0
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
