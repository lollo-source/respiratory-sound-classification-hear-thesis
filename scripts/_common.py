from __future__ import print_function

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.io_utils import root_path, write_json


def level_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--level", choices=("A", "B", "C"), required=True)
    return parser


def require_level_c(level):
    if level != "C":
        print("Level %s is prepared but not yet implemented or verified; no Level-C fallback was run." % level, file=sys.stderr)
        return False
    return True


def save_tables(tables):
    output = root_path("artifacts", "generated", "level_c")
    if not os.path.isdir(output):
        os.makedirs(output)
    for label, payload in tables.items():
        filename = label.replace("tab:", "").replace(":", "_") + ".json"
        write_json(os.path.join(output, filename), payload)
    return output

