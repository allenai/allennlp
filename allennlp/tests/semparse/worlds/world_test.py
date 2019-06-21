# pylint: disable=no-self-use,invalid-name,protected-access
from overrides import overrides

from allennlp.common.testing import AllenNlpTestCase
from allennlp.semparse import World


class FakeWorldWithoutRecursion(World):
    # pylint: disable=abstract-method
    @overrides
    def all_possible_actions(self):
        # The logical forms this grammar allows are
        # (unary_function argument)
        # (binary_function argument argument)
        actions = ['@start@ -> t',
                   't -> [<e,t>, e]',
                   '<e,t> -> unary_function',
                   '<e,t> -> [<e,<e,t>>, e]',
                   '<e,<e,t>> -> binary_function',
                   'e -> argument']
        return actions


class FakeWorldWithRecursion(FakeWorldWithoutRecursion):
    # pylint: disable=abstract-method
    @overrides
    def all_possible_actions(self):
        # In addition to the forms allowed by ``FakeWorldWithoutRecursion``, this world allows
        # (unary_function (identity .... (argument)))
        # (binary_function (identity .... (argument)) (identity .... (argument)))
        actions = super(FakeWorldWithRecursion, self).all_possible_actions()
        actions.extend(['e -> [<e,e>, e]',
                        '<e,e> -> identity'])
        return actions


class TestWorld(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.world_without_recursion = FakeWorldWithoutRecursion()
        self.world_with_recursion = FakeWorldWithRecursion()

    def test_get_paths_to_root_without_recursion(self):
        argument_paths = self.world_without_recursion.get_paths_to_root('e -> argument')
        assert argument_paths == [['e -> argument', 't -> [<e,t>, e]', '@start@ -> t'],
                                  ['e -> argument', '<e,t> -> [<e,<e,t>>, e]', 't -> [<e,t>, e]',
                                   '@start@ -> t']]
        unary_function_paths = self.world_without_recursion.get_paths_to_root('<e,t> -> unary_function')
        assert unary_function_paths == [['<e,t> -> unary_function', 't -> [<e,t>, e]',
                                         '@start@ -> t']]
        binary_function_paths = \
                self.world_without_recursion.get_paths_to_root('<e,<e,t>> -> binary_function')
        assert binary_function_paths == [['<e,<e,t>> -> binary_function',
                                          '<e,t> -> [<e,<e,t>>, e]', 't -> [<e,t>, e]',
                                          '@start@ -> t']]

    def test_get_paths_to_root_with_recursion(self):
        argument_paths = self.world_with_recursion.get_paths_to_root('e -> argument')
        # Argument now has 4 paths, and the two new paths are with the identity function occurring
        # (only once) within unary and binary functions.
        assert argument_paths == [['e -> argument', 't -> [<e,t>, e]', '@start@ -> t'],
                                  ['e -> argument', '<e,t> -> [<e,<e,t>>, e]', 't -> [<e,t>, e]',
                                   '@start@ -> t'],
                                  ['e -> argument', 'e -> [<e,e>, e]', 't -> [<e,t>, e]',
                                   '@start@ -> t'],
                                  ['e -> argument', 'e -> [<e,e>, e]', '<e,t> -> [<e,<e,t>>, e]',
                                   't -> [<e,t>, e]', '@start@ -> t']]
        identity_paths = self.world_with_recursion.get_paths_to_root('<e,e> -> identity')
        # Two identity paths, one through each of unary and binary function productions.
        assert identity_paths == [['<e,e> -> identity', 'e -> [<e,e>, e]', 't -> [<e,t>, e]',
                                   '@start@ -> t'],
                                  ['<e,e> -> identity', 'e -> [<e,e>, e]',
                                   '<e,t> -> [<e,<e,t>>, e]', 't -> [<e,t>, e]', '@start@ -> t']]
