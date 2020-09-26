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
from typing import Tuple, NamedTuple, Optional

import requests


OK_STATUS_CODES = (
    200,
    401,  # the resource exists but may require some sort of login.
    403,  # ^ same
    405,  # HEAD method not allowed.
    406,  # the resource exists, but our default 'Accept-' header may not match what the server can provide.
)

THREADS = 10

http_session = requests.Session()
for resource_prefix in ("http://", "https://"):
    http_session.mount(
        resource_prefix,
        requests.adapters.HTTPAdapter(max_retries=5, pool_connections=20, pool_maxsize=THREADS),
    )


class MatchTuple(NamedTuple):
    source: str
    name: str
    link: str


def url_ok(match_tuple: MatchTuple) -> Tuple[bool, str]:
    """Check if a URL is reachable."""
    try:
        result = http_session.head(match_tuple.link, timeout=5, allow_redirects=True)
        return (
            result.ok or result.status_code in OK_STATUS_CODES,
            f"status code = {result.status_code}",
        )
    except (requests.ConnectionError, requests.Timeout):
        return False, "connection error"


def path_ok(match_tuple: MatchTuple) -> bool:
    """Check if a file in this repository exists."""
    relative_path = match_tuple.link.split("#")[0]
    full_path = os.path.join(os.path.dirname(str(match_tuple.source)), relative_path)
    return os.path.exists(full_path)


def link_ok(match_tuple: MatchTuple) -> Tuple[MatchTuple, bool, Optional[str]]:
    reason: Optional[str] = None
    if match_tuple.link.startswith("http"):
        result_ok, reason = url_ok(match_tuple)
    else:
        result_ok = path_ok(match_tuple)
    print(f"  {'✓' if result_ok else '✗'} {match_tuple.link}")
    return match_tuple, result_ok, reason


def main():
    print("Finding all markdown files in the current directory...")

    project_root = (pathlib.Path(__file__).parent / "..").resolve()
    markdown_files = project_root.glob("**/*.md")

    all_matches = set()
    url_regex = re.compile(r"\[([^!][^\]]+)\]\(([^)(]+)\)")
    for markdown_file in markdown_files:
        with open(markdown_file) as handle:
            for line in handle.readlines():
                matches = url_regex.findall(line)
                for name, link in matches:
                    if "localhost" not in link:
                        all_matches.add(MatchTuple(source=str(markdown_file), name=name, link=link))

    print(f"  {len(all_matches)} markdown files found")
    print("Checking to make sure we can retrieve each link...")

    with Pool(processes=THREADS) as pool:
        results = pool.map(link_ok, [match for match in list(all_matches)])
    unreachable_results = [
        (match_tuple, reason) for match_tuple, success, reason in results if not success
    ]

    if unreachable_results:
        print(f"Unreachable links ({len(unreachable_results)}):")
        for match_tuple, reason in unreachable_results:
            print("  > Source: " + match_tuple.source)
            print("    Name: " + match_tuple.name)
            print("    Link: " + match_tuple.link)
            if reason is not None:
                print("    Reason: " + reason)
        sys.exit(1)
    print("No Unreachable link found.")


if __name__ == "__main__":
    main()
