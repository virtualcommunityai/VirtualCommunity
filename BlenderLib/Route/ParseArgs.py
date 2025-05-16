import sys
import argparse


def parse_script_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    return args
