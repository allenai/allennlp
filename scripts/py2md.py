#!/usr/bin/env python

"""
Turn docstring from a single module into a markdown file.
"""

import argparse
import re

from nr.databind.core import Struct
from nr.interface import implements, override
from pydoc_markdown import PydocMarkdown
from pydoc_markdown.contrib.loaders.python import PythonLoader
from pydoc_markdown.contrib.processors.crossref import CrossrefProcessor
from pydoc_markdown.contrib.renderers.markdown import MarkdownRenderer
from pydoc_markdown.interfaces import Processor, Renderer
from pydoc_markdown.reflection import Argument, Module, Function, Class


@implements(Processor)
class AllenNlpDocstringProcessor(Struct):
    """
    Use to turn our docstrings into Markdown.
    """

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
                    current_section
                    in ("arguments", "parameters", "attributes", "members", "returns")
                    and ":" in line
                ):
                    ident, ty = line.split(":", 1)
                    if ty:
                        line = f"- __{ident}__ : {ty}<br>"
                    else:
                        line = f"- __{ident}__ :<br>"
                elif current_section in ("returns", "raises"):
                    line = f"- {line} <br>"

        return line, current_section


@implements(Processor)
class AllenNlpFilterProcessor(Struct):
    """
    Used to filter out nodes that we don't want to document.
    """

    SPECIAL_MEMBERS = ("__path__", "__annotations__", "__name__", "__all__", "__init__")

    def process(self, graph, _resolver):
        graph.visit(self._process_node)

    def _process_node(self, node):
        def _check(node):
            if node.parent and node.parent.name.startswith("_"):
                return False
            if node.name.startswith("_") and not node.name.endswith("_"):
                return False
            if node.name in self.SPECIAL_MEMBERS:
                return False
            if node.name == "logger" and isinstance(node.parent, Module):
                return False
            return True

        if not _check(node):
            node.visible = False


@implements(Renderer)
class AllenNlpRenderer(MarkdownRenderer):
    def _format_function_signature(
        self, func: Function, override_name: str = None, add_method_bar: bool = True
    ) -> str:
        parts = []
        for dec in func.decorators:
            parts.append("@{}{}\n".format(dec.name, dec.args or ""))
        if self.signature_python_help_style and not func.is_method():
            parts.append("{} = ".format(func.path()))
        if func.is_async:
            parts.append("async ")
        if self.signature_with_def:
            parts.append("def ")
        if self.signature_class_prefix and (
            func.is_function() and func.parent and func.parent.is_class()
        ):
            parts.append(func.parent.name + ".")
        parts.append((override_name or func.name))
        signature_args = Argument.format_arglist(func.args)
        if signature_args.endswith(","):
            signature_args = signature_args[:-1].strip()
        if (
            len(parts[-1])
            + len(signature_args)
            + (0 if not func.return_ else len(str(func.return_)))
            > 60
        ):
            signature_args = ",\n    ".join(
                filter(lambda s: s.strip() not in ("", ","), (str(arg) for arg in func.args))
            )
            parts.append("(\n    " + signature_args + "\n)")
        else:
            parts.append("(" + signature_args + ")")
        if func.return_:
            parts.append(" -> {}".format(func.return_))
        result = "".join(parts)
        if add_method_bar and func.is_method():
            result = "\n".join(" | " + l for l in result.split("\n"))
        return result

    def _format_classdef_signature(self, cls: Class) -> str:
        bases = ", ".join(map(str, cls.bases))
        if cls.metaclass:
            bases += ", metaclass=" + str(cls.metaclass)
        code = "class {}({})".format(cls.name, bases)
        if self.signature_python_help_style:
            code = cls.path() + " = " + code
        if self.classdef_render_init_signature_if_needed and (
            "__init__" in cls.members and not cls.members["__init__"].visible
        ):
            code += ":\n" + self._format_function_signature(
                cls.members["__init__"], add_method_bar=True
            )
        return code

    def _render_module_breadcrumbs(self, fp, mod: Module):
        #  breadcrumbs = "[*allennlp*](/api)[*.modules*](/api/modules)[*.seq2seq_encoders*](/api/modules/seq2seq_encoders)[**.pytorch_transformer_wrapper**](/api/modules/seq2seq_encoders/pytorch_transformer_wrapper)"
        submods = mod.name.split(".")
        if submods[0] != "allennlp":
            return
        breadcrumbs = []
        for i, submod_name in enumerate(submods):
            if i == 0:
                title = f"*{submod_name}*"
            elif i == len(submods) - 1:
                title = f"**.{submod_name}**"
            else:
                title = f"*.{submod_name}*"
            #  href = "/api/" + "/".join(submods[1 : i + 1])
            #  breadcrumbs.append(f"[{title}]({href})")
            breadcrumbs.append(title)
        fp.write("\[ " + "".join(breadcrumbs) + " \]\n\n---\n\n")

    def _render_object(self, fp, level, obj):
        if not isinstance(obj, Module) or self.render_module_header:
            self._render_header(fp, level, obj)
        if isinstance(obj, Module):
            self._render_module_breadcrumbs(fp, obj)
        self._render_signature_block(fp, obj)
        if obj.docstring:
            lines = obj.docstring.split("\n")
            if self.docstrings_as_blockquote:
                lines = ["> " + x for x in lines]
            fp.write("\n".join(lines))
            fp.write("\n\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("module", type=str, help="The Python module to parse.")
    parser.add_argument("-o", "--out", type=str, help="Output file, default is stdout.")
    return parser.parse_args()


def main():
    opts = parse_args()

    pydocmd = PydocMarkdown(
        loaders=[PythonLoader(modules=[opts.module])],
        processors=[AllenNlpFilterProcessor(), AllenNlpDocstringProcessor(), CrossrefProcessor()],
        renderer=AllenNlpRenderer(
            filename=opts.out,
            add_method_class_prefix=False,
            add_member_class_prefix=False,
            data_code_block=True,
            signature_with_def=True,
            use_fixed_header_levels=False,
            render_module_header=False,
        ),
    )

    pydocmd.load_modules()
    pydocmd.process()
    pydocmd.render()


if __name__ == "__main__":
    main()
