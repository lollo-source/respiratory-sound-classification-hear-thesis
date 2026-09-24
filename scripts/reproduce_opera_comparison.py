#!/usr/bin/env python
from __future__ import print_function

from _common import level_parser, require_level_c, save_tables
from src.reproduction import opera_table


def main(argv=None):
    args = level_parser("Reproduce HeAR versus OPERA comparison").parse_args(argv)
    if not require_level_c(args.level):
        return 2
    table = opera_table()
    save_tables({table["label"]: table})
    print("OPERA comparison: recomputed from record predictions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

