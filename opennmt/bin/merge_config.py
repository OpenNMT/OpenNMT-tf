"""Script that merges configurations for debug or simplification."""

import argparse
import yaml

from opennmt.config import load_config


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("config", nargs="+", help="Configuration files.")
    args = parser.parse_args()
    config = load_config(args.config)
    print(yaml.dump(config, default_flow_style=False))


if __name__ == "__main__":
    main()
