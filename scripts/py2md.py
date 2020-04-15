import argparse
import re

from nr.databind.core import Struct
from nr.interface import implements, override
from pydoc_markdown import PydocMarkdown
from pydoc_markdown.contrib.processors.filter import FilterProcessor
from pydoc_markdown.contrib.processors.crossref import CrossrefProcessor
from pydoc_markdown.interfaces import Processor


@implements(Processor)
class AllenNlpMdProcessor(Struct):
    @override
    def process(self, graph, resolver):
        graph.visit(self.process_node)

    def process_node(self, node):
        if not getattr(node, "docstring", None):
            return
        lines = []
        codeblock_opened = False
        current_section = None
        for line in node.docstring.split("\n"):
            if line.startswith("```"):
                codeblock_opened = not codeblock_opened
            if not codeblock_opened:
                line, current_section = self._preprocess_line(line, current_section)
            lines.append(line)
        node.docstring = "\n".join(lines)

    def _preprocess_line(self, line, current_section):
        match = re.match(r"#+ (.*)$", line)
        if match:
            current_section = match.group(1).strip().lower()
            line = re.sub(r"#+ (.*)$", r"__\1__\n", line)
        else:
            if line and not line.startswith(" "):
                if (
                    current_section in ("arguments", "parameters", "attributes", "members")
                    and ":" in line
                ):
                    ident, ty = line.split(":", 1)
                    if ty:
                        line = f"- __{ident}__ : {ty}<br>"
                    else:
                        line = f"- __{ident}__ :<br>"
                elif current_section in ("returns", "raises"):
                    line = f"{line}<br>"

        return line, current_section


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("module", type=str, help="The Python module to parse.")
    parser.add_argument("-o", "--out", type=str, help="Output file, default is stdout.")
    return parser.parse_args()


def main():
    opts = parse_args()

    pydocmd = PydocMarkdown()
    pydocmd.loaders[0].modules = [opts.module]
    pydocmd.processors = [FilterProcessor(), AllenNlpMdProcessor(), CrossrefProcessor()]
    if opts.out:
        pydocmd.renderer.filename = opts.out

    pydocmd.load_modules()
    pydocmd.process()
    pydocmd.render()


if __name__ == "__main__":
    main()
