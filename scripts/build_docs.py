import os
from pathlib import Path
from subprocess import check_output
from typing import Dict, List, Any

from ruamel.yaml import YAML

parent_folder_path = Path(__file__).parent.parent

yaml_path = parent_folder_path / "mkdocs.yml"
source_path = parent_folder_path / "allennlp"

docs_dir = str(parent_folder_path / "docs" / "api")
yaml = YAML()


print("Building API docs...")
exclude_files = [".DS_Store", "__init__.py", "__init__.pyc", "README.md", "version.py", "run.py"]


def render_docs(src_rel_path: str, src_file: str, to_file: str, modifier="++"):
    src_rel_ns = src_rel_path.replace("/", ".")
    src_base = src_file.replace(".py", "")

    if src_rel_ns == "":
        namespace = f"allennlp.{src_base}{modifier}"
    else:
        namespace = f"allennlp.{src_rel_ns}.{src_base}{modifier}"

    args = ["pydocmd", "simple", namespace]
    call_result = check_output(args, env=os.environ).decode("utf-8")
    with open(to_file, "w") as file:
        file.write(call_result)

    print(f"Built docs for {src_file}: {to_file}")


def build_docs_for_file(relative_path: str, file_name: str, docs_dir: str):

    clean_filename = file_name.replace(".py", "")
    markdown_filename = f"{clean_filename}.md"

    output_path = os.path.join(docs_dir, relative_path, markdown_filename)
    nav_path = os.path.join("api", relative_path, markdown_filename)
    nav_item: Any = dict()
    nav_item[os.path.basename(clean_filename)] = nav_path

    render_docs(relative_path, file_name, output_path)

    return nav_item


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
            nav_root.append({child: nav_subsection})

        else:
            if child in exclude_files or not child.endswith(".py"):
                continue

            x = build_docs_for_file(without_allennlp, child, docs_dir)
            nav_root.append(x)

    return nav_root


if __name__ == "__main__":
    print(f"Render to: {docs_dir}")

    nav_entries = build_docs(source_path, docs_dir)
    YAMLSection = List[Dict[str, List[Dict[str, str]]]]

    nav_entries.sort(key=lambda x: list(x)[0], reverse=False)

    mkdocs_yaml = yaml.load(yaml_path)
    docs_key = "API"
    site_nav = mkdocs_yaml["nav"]
    for nav_obj in site_nav:
        if docs_key in nav_obj:
            nav_obj[docs_key] = nav_entries
            break

    out = mkdocs_yaml
    with open(yaml_path, "w") as f:
        yaml.dump(mkdocs_yaml, f)
    print("done!")
