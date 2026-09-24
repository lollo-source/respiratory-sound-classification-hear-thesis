#!/usr/bin/env python
from __future__ import print_function

from _common import level_parser, require_level_c, save_tables
from src.reproduction import bootstrap_tables


def main(argv=None):
    args = level_parser("Reproduce patient-cluster bootstrap tables").parse_args(argv)
    if not require_level_c(args.level):
        return 2
    tables = bootstrap_tables()
    save_tables(tables)
    print("bootstrap: recomputed 10,000 paired patient-cluster replicates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

