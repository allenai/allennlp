from mathy_pydoc.__main__ import main as mathy_pydoc_main
from contextlib import redirect_stdout
import sys
from typing import Dict
import os
from pathlib import Path

from ruamel.yaml import YAML


exclude_files = [
    ".DS_Store",
    "__init__.py",
    "__init__.pyc",
    "README.md",
    "version.py",
    "__main__.py",
]


def render_file(relative_src_path: str, src_file: str, to_file: str, modifier="++") -> None:
    """
    Shells out to pydocmd, which creates a .md file from the docstrings of python functions and classes in
    the file we specify. The modifer specifies the depth at which to generate docs for classes and functions
    in the file. More information here: https://pypi.org/project/pydoc-markdown/

    """
    relative_src_namespace = relative_src_path.replace("/", ".")
    src_base = src_file.replace(".py", "")

    if relative_src_namespace == "":
        namespace = f"allennlp.{src_base}{modifier}"
    else:
        namespace = f"allennlp.{relative_src_namespace}.{src_base}{modifier}"

    sys.argv = ["mathy_pydoc", namespace]
    with open(to_file, "w") as f:
        with redirect_stdout(f):
            mathy_pydoc_main()

    print(f"Built docs for {src_file}: {to_file}")
    print()


def build_docs_for_file(relative_path: str, file_name: str, docs_dir: str) -> Dict[str, str]:
    """
    Build docs for an individual python file.
    """
    clean_filename = file_name.replace(".py", "")
    markdown_filename = f"{clean_filename}.md"

    output_path = os.path.join(docs_dir, relative_path, markdown_filename)
    nav_path = os.path.join("api", relative_path, markdown_filename)
    render_file(relative_path, file_name, output_path)

    return {os.path.basename(clean_filename): nav_path}


def build_docs(root_path: str, docs_dir: str):

    nav_root = []

    for child in os.listdir(root_path):
        relative_path = os.path.join(root_path, child)

        if (
            "__pycache__" in relative_path
            or "tests" in relative_path
            or "mypy_cache" in relative_path
        ):
            continue

        without_allennlp = str(root_path).replace("allennlp/", "")
        target_dir = os.path.join(docs_dir, without_allennlp)
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)

        if os.path.isdir(relative_path):
            nav_subsection = build_docs(relative_path, docs_dir)
            if not nav_subsection:
                continue
            nav_subsection.sort(key=lambda x: list(x)[0], reverse=False)
            nav_root.append({child: nav_subsection})

        else:
            if child in exclude_files or not child.endswith(".py"):
                continue

            nav = build_docs_for_file(without_allennlp, child, docs_dir)
            nav_root.append(nav)

    return nav_root


if __name__ == "__main__":

    print("Building the docs.")
    parent_folder_path = Path(__file__).parent.parent
    yaml_path = parent_folder_path / "mkdocs.yml"
    source_path = parent_folder_path / "allennlp"
    docs_dir = str(parent_folder_path / "docs" / "api")
    if not os.path.exists(docs_dir):
        os.mkdir(docs_dir)
    yaml = YAML()

    nav_entries = build_docs(source_path, docs_dir)
    nav_entries.sort(key=lambda x: list(x)[0], reverse=False)

    mkdocs_yaml = yaml.load(yaml_path)
    docs_key = "API"
    site_nav = mkdocs_yaml["nav"]

    # Find the yaml corresponding to the API
    for nav_obj in site_nav:
        if docs_key in nav_obj:
            break

    nav_obj[docs_key] = nav_entries

    with open(yaml_path, "w") as f:
        yaml.dump(mkdocs_yaml, f)
