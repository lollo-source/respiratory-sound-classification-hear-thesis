#!/usr/bin/env python
from __future__ import print_function

from _common import level_parser, require_level_c, save_tables
from src.reproduction import progression_tables


def main(argv=None):
    args = level_parser("Reproduce progression tables").parse_args(argv)
    if not require_level_c(args.level):
        return 2
    tables = progression_tables()
    save_tables(tables)
    print("progression: wrote %d independently computed tables" % len(tables))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

