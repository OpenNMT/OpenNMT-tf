"""Standalone script to detokenize a corpus."""

import argparse

from opennmt import tokenizers


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--delimiter", default=" ", help="Token delimiter used in text serialization."
    )
    parser.add_argument(
        "--tokenizer_config", default=None, help="Tokenization configuration."
    )
    args = parser.parse_args()

    tokenizer = tokenizers.make_tokenizer(args.tokenizer_config)
    tokenizer.detokenize_stream(delimiter=args.delimiter)


if __name__ == "__main__":
    main()
