#!/usr/bin/env python
"""Verify an explicit Quick or Full public results bundle."""

from __future__ import print_function

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.results_verification import verify_bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description="Verify a generated public results bundle")
    parser.add_argument("--mode", choices=("quick", "full"), required=True)
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args(argv)
    try:
        result = verify_bundle(args.results_dir, ROOT, mode=args.mode)
    except Exception as error:
        print("Results verification stopped: %s" % error, file=sys.stderr)
        return 1
    return 0 if result["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
