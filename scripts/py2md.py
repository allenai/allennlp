#!/usr/bin/env python

"""
Turn docstrings from a single module into a markdown file.

We do this with PydocMarkdown, using custom processors and renderers defined here.
"""

import argparse
from collections import deque, OrderedDict
from dataclasses import dataclass
from enum import Enum
import logging
from multiprocessing import Pool, cpu_count
import os
from pathlib import Path
import re
from typing import Optional, Tuple, List

from nr.databind.core import Struct
from nr.interface import implements, override
from pydoc_markdown import PydocMarkdown
from pydoc_markdown.contrib.loaders.python import PythonLoader
from pydoc_markdown.contrib.renderers.markdown import MarkdownRenderer
from pydoc_markdown.interfaces import Processor, Renderer
from pydoc_markdown.reflection import Argument, Module, Function, Class


logging.basicConfig(level=logging.INFO)


class Section(Enum):
    ARGUMENTS = "ARGUMENTS"
    PARAMETERS = "PARAMETERS"
    ATTRIBUTES = "ATTRIBUTES"
    MEMBERS = "MEMBERS"
    RETURNS = "RETURNS"
    RAISES = "RAISES"
    EXAMPLES = "EXAMPLES"
    OTHER = "OTHER"

    @classmethod
    def from_str(cls, section: str) -> "Section":
        section = section.upper()
        for member in cls:
            if section == member.value:
                return member
        return cls.OTHER


@dataclass
class Param:
    ident: str
    ty: Optional[str] = None
    required: bool = False
    default: Optional[str] = None

    @classmethod
    def from_line(cls, line: str) -> Optional["Param"]:
        if ":" not in line:
            return None
        ident, description = line.split(":", 1)
        ident = ident.strip()
        description = description.strip()
        ty = None
        required = True
        default = None
        if "`, " in description:
            ty, extras = description.split("`, ", 1)
            ty = ty + "`"
            required = "optional" not in extras
            default_match = re.match(r".*default = (`?[^\s`\)]+`?).*", extras)
            if default_match:
                default = default_match.group(1)
                if not default.startswith("`"):
                    logging.warning("Default should be enclosed in backticks: '%s'", line)
        else:
            ty = description
        if not ty.startswith("`"):
            logging.warning("Type should be enclosed in backticks: '%s'", line)
        return cls(ident=ident, ty=ty, required=required, default=default)

    def to_line(self) -> str:
        line: str = f"- __{self.ident}__ :"
        if self.ty:
            line += f" {self.ty}"
        if not self.required:
            line += ", optional"
            if self.default:
                line += f" (default = {self.default})"
        line += "<br>"
        return line


# For now we handle attributes / members in the same way as parameters / arguments.
Attrib = Param


@dataclass
class RetVal:
    description: Optional[str] = None
    ident: Optional[str] = None
    ty: Optional[str] = None

    @classmethod
    def from_line(cls, line: str) -> "RetVal":
        if ":" not in line:
            return cls(description=line)
        ident, ty = line.split(":", 1)
        ident = ident.strip()
        ty = ty.strip()
        if ty and not ty.startswith("`"):
            logging.warning("Type should be enclosed in backticks: '%s'", line)
        return cls(ident=ident, ty=ty)

    def to_line(self) -> str:
        if self.description:
            line = f"- {self.description} <br>"
        elif self.ident:
            line = f"- __{self.ident}__"
            if self.ty:
                line += f" : {self.ty}<br>"
            else:
                line += "<br>"
        else:
            raise TypeError("RetVal must have either description or ident")
        return line


@dataclass
class ProcessorState:
    parameters: "OrderedDict[str, Param]"
    current_section: Optional[Section] = None
    codeblock_opened: bool = False
    consecutive_blank_line_count: int = 0


@implements(Processor)
class AllenNlpDocstringProcessor(Struct):
    """
    Use to turn our docstrings into Markdown.
    """

    CROSS_REF_RE = re.compile("(:(class|func|mod):`~?([a-zA-Z0-9_.]+)`)")

    @override
    def process(self, graph, resolver):
        graph.visit(self.process_node)

    def process_node(self, node):
        if not getattr(node, "docstring", None):
            return

        lines: List[str] = []
        state: ProcessorState = ProcessorState(parameters=OrderedDict())

        for line in node.docstring.split("\n"):
            # Check if we're starting or ending a codeblock.
            if line.startswith("```"):
                state.codeblock_opened = not state.codeblock_opened

            if not state.codeblock_opened:
                # If we're not in a codeblock, we'll do some pre-processing.
                if not line.strip():
                    state.consecutive_blank_line_count += 1
                    if state.consecutive_blank_line_count >= 2:
                        state.current_section = None
                else:
                    state.consecutive_blank_line_count = 0
                line = self._preprocess_line(line, state)

            lines.append(line)

        # Now set the docstring to our preprocessed version of it.
        node.docstring = "\n".join(lines)

    def _preprocess_line(self, line, state: ProcessorState) -> str:
        match = re.match(r"#+ (.*)$", line)
        if match:
            state.current_section = Section.from_str(match.group(1).strip())
            line = re.sub(r"#+ (.*)$", r"__\1__\n", line)
        else:
            if line and not line.startswith(" ") and not line.startswith("!!! "):
                if state.current_section in (Section.ARGUMENTS, Section.PARAMETERS,):
                    param = Param.from_line(line)
                    if param:
                        line = param.to_line()
                elif state.current_section in (Section.ATTRIBUTES, Section.MEMBERS):
                    attrib = Attrib.from_line(line)
                    if attrib:
                        line = attrib.to_line()
                elif state.current_section in (Section.RETURNS, Section.RAISES):
                    retval = RetVal.from_line(line)
                    line = retval.to_line()

            line = self._transform_cross_references(line)

        return line

    def _transform_cross_references(self, line: str) -> str:
        """
        Replace sphinx style crossreferences with markdown links.
        """
        for match, ty, name in self.CROSS_REF_RE.findall(line):
            if name.startswith("allennlp."):
                path = name.split(".")
                if ty == "mod":
                    href = "/api/" + "/".join(path[1:])
                else:
                    href = "/api/" + "/".join(path[1:-1]) + "#" + path[-1].lower()
                cross_ref = f"[`{path[-1]}`]({href})"
            elif "." not in name:
                cross_ref = f"[`{name}`](#{name.lower()})"
            else:
                cross_ref = f"`{name}`"
            line = line.replace(match, cross_ref)
        return line


@implements(Processor)
class AllenNlpFilterProcessor(Struct):
    """
    Used to filter out nodes that we don't want to document.
    """

    def process(self, graph, _resolver):
        graph.visit(self._process_node)

    def _process_node(self, node):
        def _check(node):
            if node.parent and node.parent.name.startswith("_"):
                return False
            if node.name.startswith("_"):
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
            result = "\n".join(" | " + line for line in result.split("\n"))
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
        fp.write("[ " + "".join(breadcrumbs) + " ]\n\n---\n\n")

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


def py2md(module: str, out: Optional[str] = None) -> None:
    pydocmd = PydocMarkdown(
        loaders=[PythonLoader(modules=[module])],
        processors=[AllenNlpFilterProcessor(), AllenNlpDocstringProcessor()],
        renderer=AllenNlpRenderer(
            filename=out,
            add_method_class_prefix=False,
            add_member_class_prefix=False,
            data_code_block=True,
            signature_with_def=True,
            use_fixed_header_levels=False,
            render_module_header=False,
        ),
    )
    if out:
        out_path = Path(out)
        os.makedirs(out_path.parent, exist_ok=True)

    pydocmd.load_modules()
    pydocmd.process()
    pydocmd.render()
    logging.info("Processed %s", module)


def _py2md_wrapper(x: Tuple[str, str]):
    """
    Used to wrap py2md since we can't pickle a lambda (needed for multiprocessing).
    """
    return py2md(x[0], x[1])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("modules", nargs="+", type=str, help="""The Python modules to parse.""")
    parser.add_argument(
        "-o",
        "--out",
        nargs="+",
        type=str,
        help="""Output files.
                If given, must have the same number of items as 'modules'.
                If not given, stdout is used.""",
    )
    return parser.parse_args()


def main():
    opts = parse_args()
    outputs = opts.out if opts.out else [None] * len(opts.modules)
    if len(outputs) != len(opts.modules):
        raise ValueError("Number inputs and outputs should be the same.")
    n_threads = cpu_count()
    if len(opts.modules) > n_threads and opts.out:
        # If writing to files, can process in parallel.
        chunk_size = max([1, int(len(outputs) / n_threads)])
        logging.info("Using %d threads", n_threads)
        with Pool(n_threads) as p:
            deque(p.imap(_py2md_wrapper, zip(opts.modules, outputs), chunk_size), maxlen=0)
    else:
        # If writing to stdout, need to process sequentially. Otherwise the output
        # could get intertwined.
        for module, out in zip(opts.modules, outputs):
            py2md(module, out)
    logging.info("Processed %d modules", len(opts.modules))


if __name__ == "__main__":
    main()
