#!/usr/bin/env python

"""
This script is used to populate the table of contents for the API in the mkdocs config file.
"""

import argparse
from pathlib import Path
from typing import Any, List

from ruamel.yaml import YAML

from allennlp.version import VERSION


API_TOC_KEY = "API"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("target_yaml", help="Path to the target mkdocs config file.")
    parser.add_argument("source_yaml", help="Path to the mkdocs skeleton config file.")
    parser.add_argument("docs_root", help="The root of the markdown docs folder.")
    parser.add_argument(
        "api_docs_path", help="The root of the API docs within the markdown docs root folder."
    )
    parser.add_argument("--docs-version", type=str, default=f"v{VERSION}")
    return parser.parse_args()


def build_api_toc(source_path: Path, docs_root: Path):
    nav_entries: List[Any] = []

    for child in source_path.iterdir():
        if child.is_dir():
            nav_subsection = build_api_toc(child, docs_root)
        elif child.suffix == ".md":
            nav_subsection = str(child.relative_to(docs_root))
        nav_entries.append({child.stem: nav_subsection})

    nav_entries.sort(key=lambda x: list(x)[0], reverse=False)
    return nav_entries


def main():
    yaml = YAML()
    opts = parse_args()

    source_yaml = yaml.load(Path(opts.source_yaml))

    nav_entries = build_api_toc(Path(opts.api_docs_path), Path(opts.docs_root))

    # Add version to name.
    source_yaml["site_name"] = f"AllenNLP {opts.docs_version}"

    # Find the yaml sub-object corresponding to the API table of contents.
    site_nav = source_yaml["nav"]
    for nav_obj in site_nav:
        if API_TOC_KEY in nav_obj:
            break
    nav_obj[API_TOC_KEY] = nav_entries

    with open(opts.target_yaml, "w") as f:
        yaml.dump(source_yaml, f)

    print(f"{opts.target_yaml} created")


if __name__ == "__main__":
    main()
