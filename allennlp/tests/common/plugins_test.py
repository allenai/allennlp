from overrides import overrides

from allennlp.commands import Subcommand
from allennlp.common.plugins import (
    discover_plugins,
    import_plugins,
)
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import pushd


class TestPlugins(AllenNlpTestCase):
    @overrides
    def setUp(self):
        super().setUp()
        self.plugins_root = self.FIXTURES_ROOT / "plugins"

    def test_no_plugins(self):
        available_plugins = set(discover_plugins())
        self.assertSetEqual(set(), available_plugins)

    def test_file_plugin(self):
        available_plugins = set(discover_plugins())
        self.assertSetEqual(set(), available_plugins)

        with pushd(self.plugins_root):
            available_plugins = set(discover_plugins())
            self.assertSetEqual({"d"}, available_plugins)

            import_plugins()
            subcommands_available = Subcommand.list_available()
            self.assertIn("d", subcommands_available)
