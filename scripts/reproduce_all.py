#!/usr/bin/env python
from __future__ import print_function

import os

from _common import level_parser, require_level_c, save_tables
from src.io_utils import write_json
from src.reproduction import progression_tables, opera_table, challenge_distribution_table, challenge_tables, bootstrap_tables, LABELS


def main(argv=None):
    args = level_parser("Reproduce every thesis table").parse_args(argv)
    if not require_level_c(args.level):
        return 2
    tables = {}
    tables.update(progression_tables())
    opera = opera_table()
    tables[opera["label"]] = opera
    distribution = challenge_distribution_table()
    tables[distribution["label"]] = distribution
    tables.update(challenge_tables())
    tables.update(bootstrap_tables())
    missing = sorted(set(LABELS) - set(tables))
    if missing:
        raise RuntimeError("missing generated tables: %s" % missing)
    output = save_tables(tables)
    write_json(os.path.join(output, "index.json"), {"level": "C", "table_count": len(tables), "labels": LABELS})
    print("Level C complete: %d independently computed thesis tables in %s" % (len(tables), os.path.relpath(output)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

