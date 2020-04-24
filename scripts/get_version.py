#!/usr/bin/env python

VERSION = {}
with open("allennlp/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

print(VERSION["VERSION"])
