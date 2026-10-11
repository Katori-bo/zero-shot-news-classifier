"""Command-line entry point: python cli.py --help."""
import argparse
import json
import sys
from pathlib import Path

from classifier import NewsClassifier


def main():
    parser = argparse.ArgumentParser(description="Classify English text without target-label training.")
    parser.add_argument("text", help="Passage enclosed in quotes")
    parser.add_argument("--labels", nargs="+", default=["sports", "technology", "politics", "education"])
    parser.add_argument("--multi-label", action="store_true", help="Score categories independently")
    parser.add_argument("--device", type=int, default=-1, help="-1 for CPU; 0 for the first CUDA GPU")
    parser.add_argument("--revision", help="Optional Hugging Face model commit SHA")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()
    try:
        result = NewsClassifier(device=args.device, revision=args.revision).classify(
            args.text, args.labels, multi_label=args.multi_label,
        )
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    output = json.dumps(result, indent=2)
    print(output)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
