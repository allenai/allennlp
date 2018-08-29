#!/usr/bin/env python
# encoding: UTF-8

"""
Goes through all the inline-links in markdown files and reports the breakages.
"""

import re
import sys
import pathlib
import os
from multiprocessing.dummy import Pool
from typing import Tuple, NamedTuple

import requests


class MatchTuple(NamedTuple):
    source: str
    name: str
    link: str


def url_ok(match_tuple: MatchTuple) -> bool:
    """Check if a URL is reachable."""
    try:
        result = requests.get(match_tuple.link, timeout=5)
        return result.ok
    except (requests.ConnectionError, requests.Timeout):
        return False


def path_ok(match_tuple: MatchTuple) -> bool:
    """Check if a file in this repository exists."""
    relative_path = match_tuple.link.split("#")[0]
    full_path = os.path.join(os.path.dirname(str(match_tuple.source)), relative_path)
    return os.path.exists(full_path)


def link_ok(match_tuple: MatchTuple) -> Tuple[MatchTuple, bool]:
    if match_tuple.link.startswith("http"):
        result_ok = url_ok(match_tuple)
    else:
        result_ok = path_ok(match_tuple)
    print(f"  {'✓' if result_ok else '✗'} {match_tuple.link}")
    return match_tuple, result_ok


def main():
    print("Finding all markdown files in the current directory...")

    project_root = (pathlib.Path(__file__).parent / "..").resolve() # pylint: disable=no-member
    markdown_files = project_root.glob('**/*.md')

    all_matches = set()
    url_regex = re.compile(r'\[([^!][^\]]+)\]\(([^)(]+)\)')
    for markdown_file in markdown_files:
        with open(markdown_file) as handle:
            for line in handle.readlines():
                matches = url_regex.findall(line)
                for name, link in matches:
                    if 'localhost' not in link:
                        all_matches.add(MatchTuple(source=str(markdown_file), name=name, link=link))

    print(f"  {len(all_matches)} markdown files found")
    print("Checking to make sure we can retrieve each link...")

    with Pool(processes=10) as pool:
        results = pool.map(link_ok, [match for match in list(all_matches)])
    unreachable_results = [match_tuple for match_tuple, success in results if not success]

    if unreachable_results:
        print(f"Unreachable links ({len(unreachable_results)}):")
        for match_tuple in unreachable_results:
            print("  > Source: " + match_tuple.source)
            print("    Name: " + match_tuple.name)
            print("    Link: " + match_tuple.link)
        sys.exit(1)
    print("No Unreachable link found.")


if __name__ == "__main__":
    main()
