#!/usr/bin/env python

import argparse
from typing import Dict

import requests


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("version_type", choices=["stable", "latest", "current"])
    return parser.parse_args()


def get_current_version() -> str:
    VERSION: Dict[str, str] = {}
    with open("allennlp/version.py", "r") as version_file:
        exec(version_file.read(), VERSION)
    return "v" + VERSION["VERSION"]


def get_latest_version() -> str:
    resp = requests.get("https://api.github.com/repos/allenai/allennlp/tags")
    return resp.json()[0]["name"]


def get_stable_version() -> str:
    resp = requests.get("https://api.github.com/repos/allenai/allennlp/releases/latest")
    return resp.json()["tag_name"]


def main() -> None:
    opts = parse_args()
    if opts.version_type == "stable":
        print(get_stable_version())
    elif opts.version_type == "latest":
        print(get_latest_version())
    elif opts.version_type == "current":
        print(get_current_version())
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
