#!/usr/bin/env python

"""
Turn docstrings from a single module into a markdown file.

We do this with PydocMarkdown, using custom processors and renderers defined here.
"""

import argparse
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
import logging
from multiprocessing import Pool, cpu_count
import os
from pathlib import Path
import re
import sys
from typing import Optional, Tuple, List

from nr.databind.core import Struct
from nr.interface import implements, override
from pydoc_markdown import PydocMarkdown
from pydoc_markdown.contrib.loaders.python import PythonLoader
from pydoc_markdown.contrib.renderers.markdown import MarkdownRenderer
from pydoc_markdown.interfaces import Processor, Renderer
from pydoc_markdown.reflection import Argument, Module, Function, Class, Data


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("py2md")


class DocstringError(Exception):
    pass


def emphasize(s: str) -> str:
    # Need to escape underscores.
    s = s.replace("_", "\\_")
    return f"__{s}__"


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


REQUIRED_PARAM_RE = re.compile(r"^`([^`]+)`(, required\.?)?$")

OPTIONAL_PARAM_RE = re.compile(
    r"^`([^`]+)`,?\s+(optional,?\s)?\(\s?(optional,\s)?default\s?=\s?`([^`]+)`\s?\)\.?$"
)

OPTIONAL_PARAM_NO_DEFAULT_RE = re.compile(r"^`([^`]+)`,?\s+optional\.?$")


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

        if " " in ident:
            return None

        maybe_match = REQUIRED_PARAM_RE.match(description)
        if maybe_match:
            ty = maybe_match.group(1)
            return cls(ident=ident, ty=ty, required=True)

        maybe_match = OPTIONAL_PARAM_RE.match(description)
        if maybe_match:
            ty = maybe_match.group(1)
            default = maybe_match.group(4)
            return cls(ident=ident, ty=ty, required=False, default=default)

        maybe_match = OPTIONAL_PARAM_NO_DEFAULT_RE.match(description)
        if maybe_match:
            ty = maybe_match.group(1)
            return cls(ident=ident, ty=ty, required=False)

        raise DocstringError(
            f"Invalid parameter / attribute description: '{line}'\n"
            "Make sure types are enclosed in backticks.\n"
            "Required parameters should be documented like: '{ident} : `{type}`'\n"
            "Optional parameters should be documented like: '{ident} : `{type}`, optional (default = `{expr}`)'\n"
        )

    def to_line(self) -> str:
        line: str = f"- {emphasize(self.ident)} :"
        if self.ty:
            line += f" `{self.ty}`"
            if not self.required:
                line += ", optional"
                if self.default:
                    line += f" (default = `{self.default}`)"
        line += " <br>"
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
        if ": " not in line:
            return cls(description=line)
        ident, ty = line.split(":", 1)
        ident = ident.strip()
        ty = ty.strip()
        if ty and not ty.startswith("`"):
            raise DocstringError(f"Type should be enclosed in backticks: '{line}'")
        return cls(ident=ident, ty=ty)

    def to_line(self) -> str:
        if self.description:
            line = f"- {self.description} <br>"
        elif self.ident:
            line = f"- {emphasize(self.ident)}"
            if self.ty:
                line += f" : {self.ty} <br>"
            else:
                line += " <br>"
        else:
            raise DocstringError("RetVal must have either description or ident")
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
    UNDERSCORE_HEADER_RE = re.compile(r"(.*)\n-{3,}\n")
    MULTI_LINE_LINK_RE = re.compile(r"(\[[^\]]+\])\n\s*(\([^\)]+\))")

    @override
    def process(self, graph, resolver):
        graph.visit(self.process_node)

    def process_node(self, node):
        if not getattr(node, "docstring", None):
            return

        lines: List[str] = []
        state: ProcessorState = ProcessorState(parameters=OrderedDict())

        docstring = node.docstring

        # Standardize header syntax to use '#' instead of underscores.
        docstring = self.UNDERSCORE_HEADER_RE.sub(r"# \g<1>", docstring)

        # It's common to break up markdown links into multiple lines in docstrings, but
        # they won't render as links in the doc HTML unless they are all on one line.
        docstring = self.MULTI_LINE_LINK_RE.sub(r"\g<1>\g<2>", docstring)

        for line in docstring.split("\n"):
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
            line = re.sub(r"#+ (.*)$", r"<strong>\1</strong>\n", line)
        else:
            if line and not line.startswith(" ") and not line.startswith("!!! "):
                if state.current_section in (
                    Section.ARGUMENTS,
                    Section.PARAMETERS,
                ):
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

    PRIVATE_METHODS_TO_KEEP = {"DatasetReader._read", "__call__"}

    def process(self, graph, _resolver):
        graph.visit(self._process_node)

    def _process_node(self, node):
        def _check(node):
            if node.name.startswith("_"):
                if node.name in self.PRIVATE_METHODS_TO_KEEP:
                    return True
                if (
                    node.parent
                    and f"{node.parent.name}.{node.name}" in self.PRIVATE_METHODS_TO_KEEP
                ):
                    return True
                return False
            if node.parent and node.parent.name.startswith("_"):
                return False
            if node.name == "logger" and isinstance(node.parent, Module):
                return False
            return True

        if not _check(node):
            node.visible = False


@implements(Renderer)
class AllenNlpRenderer(MarkdownRenderer):
    def _format_function_signature(
        self,
        func: Function,
        override_name: str = None,
        add_method_bar: bool = True,
        include_parent_class: bool = True,
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
            if include_parent_class:
                bases = ", ".join(map(str, func.parent.bases))
                if func.parent.metaclass:
                    bases += ", metaclass=" + str(func.parent.metaclass)
                if bases:
                    class_signature = f"class {func.parent.name}({bases})"
                else:
                    class_signature = f"class {func.parent.name}"
                result = f"{class_signature}:\n | ...\n{result}"
        return result

    def _format_data_signature(self, data: Data) -> str:
        expr = str(data.expr)
        if len(expr) > self.data_expression_maxlength:
            expr = expr[: self.data_expression_maxlength] + " ..."

        if data.annotation:
            signature = f"{data.name}: {data.annotation} = {expr}"
        else:
            signature = f"{data.name} = {expr}"

        if data.parent and data.parent.is_class():
            bases = ", ".join(map(str, data.parent.bases))
            if data.parent.metaclass:
                bases += ", metaclass=" + str(data.parent.metaclass)
            if bases:
                class_signature = f"class {data.parent.name}({bases})"
            else:
                class_signature = f"class {data.parent.name}"
            return f"{class_signature}:\n | ...\n | {signature}"
        else:
            return signature

    def _format_classdef_signature(self, cls: Class) -> str:
        code = ""
        if cls.decorators:
            for dec in cls.decorators:
                code += "@{}{}\n".format(dec.name, dec.args or "")
        bases = ", ".join(map(str, cls.bases))
        if cls.metaclass:
            bases += ", metaclass=" + str(cls.metaclass)
        if bases:
            code += "class {}({})".format(cls.name, bases)
        else:
            code += "class {}".format(cls.name)
        if self.signature_python_help_style:
            code = cls.path() + " = " + code
        if self.classdef_render_init_signature_if_needed and (
            "__init__" in cls.members and not cls.members["__init__"].visible
        ):
            code += ":\n" + self._format_function_signature(
                cls.members["__init__"],
                add_method_bar=True,
                include_parent_class=False,
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


def py2md(module: str, out: Optional[str] = None) -> bool:
    """
    Returns `True` if module successfully processed, otherwise `False`.
    """
    logger.debug("Processing %s", module)
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
            descriptive_class_title=False,
        ),
    )
    if out:
        out_path = Path(out)
        os.makedirs(out_path.parent, exist_ok=True)

    pydocmd.load_modules()
    try:
        pydocmd.process()
    except DocstringError as err:
        logger.exception("Failed to process %s.\n%s", module, err)
        return False
    pydocmd.render()
    return True


def _py2md_wrapper(x: Tuple[str, str]) -> bool:
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
    errors: int = 0
    if len(opts.modules) > n_threads and opts.out:
        # If writing to files, can process in parallel.
        chunk_size = max([1, int(len(outputs) / n_threads)])
        logger.info("Using %d threads", n_threads)
        with Pool(n_threads) as p:
            for result in p.imap(_py2md_wrapper, zip(opts.modules, outputs), chunk_size):
                if not result:
                    errors += 1
    else:
        # If writing to stdout, need to process sequentially. Otherwise the output
        # could get intertwined.
        for module, out in zip(opts.modules, outputs):
            result = py2md(module, out)
            if not result:
                errors += 1
    logger.info("Processed %d modules", len(opts.modules))
    if errors:
        logger.error("Found %d errors", errors)
        sys.exit(1)


if __name__ == "__main__":
    main()
