# pylint: disable=no-self-use
from allennlp.common.testing import AllenNlpTestCase
from allennlp.semparse.type_declarations import type_declaration as base_types

from allennlp.semparse.type_declarations.ratecalculus_type_declaration import (
        CONJUNCTION_TYPE,
        BOOLEAN_TYPE
        )


class RateCalculusTypeDeclarationTest(AllenNlpTestCase):

    def test_conjunction_maps_to_correct_actions(self):
        valid_actions = base_types.get_valid_actions({'And': 'O'},
                                                     {'O': CONJUNCTION_TYPE},
                                                     {BOOLEAN_TYPE},
                                                     {BOOLEAN_TYPE})
        assert 'b -> [<b,<b,b>>, b, b]' in valid_actions['b']
