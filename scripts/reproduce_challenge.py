#!/usr/bin/env python
from __future__ import print_function

from _common import level_parser, require_level_c, save_tables
from src.reproduction import challenge_distribution_table, challenge_tables


def main(argv=None):
    args = level_parser("Reproduce BioCAS Challenge tables").parse_args(argv)
    if not require_level_c(args.level):
        return 2
    distribution = challenge_distribution_table()
    tables = challenge_tables()
    tables[distribution["label"]] = distribution
    save_tables(tables)
    print("Challenge: recomputed component scores and retrospective placements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

