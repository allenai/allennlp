#!/usr/bin/env python

import os
from subprocess import run

value = os.environ.get("BUILD_DEMO", "false")
if value.lower() == "true":
    run("npm install", shell=True, check=True, cwd="demo")
    run("npm run build", shell=True, check=True, cwd="demo")
    print("Demo built")
else:
    print("BUILD_DEMO is '%s'.  Not building demo." % value)
