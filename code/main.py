"""End-to-end support triage pipeline entry point."""

from __future__ import annotations

import argparse

from config import OUTPUT_CSV, REPO_ROOT
from pipeline import Pipeline
from scorer import print_score_report, score
from utils.secrets_validator import validate


def build_parser() -> argparse.ArgumentParser:
    default_input = REPO_ROOT / "support_tickets" / "support_tickets.csv"
    parser = argparse.ArgumentParser(
        description="Support Triage Agent — processes support tickets and routes to triage status.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Full run on support_tickets/support_tickets.csv
  python main.py --sample           # Run on sample CSV with internal scoring
  python main.py --dry-run          # Process first 5 tickets only
  python main.py --input custom.csv # Process custom CSV input
        """,
    )
    parser.add_argument(
        "--input", default=str(default_input), help="Input CSV path"
    )
    parser.add_argument("--output", default=OUTPUT_CSV, help="Output CSV path")
    parser.add_argument(
        "--sample", action="store_true", help="Run on sample CSV and compute internal scores"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Process only first 5 tickets for testing"
    )
    return parser


def main() -> int:
    """Main entry point for triage pipeline."""
    validate()

    parser = build_parser()
    args = parser.parse_args()

    input_csv = args.input
    output_csv = args.output

    if args.sample:
        input_csv = str(REPO_ROOT / "support_tickets" / "sample_support_tickets.csv")
        output_csv = str(REPO_ROOT / "support_tickets" / "output.csv")

    max_rows = 5 if args.dry_run else None

    pipeline = Pipeline()
    _stats, _results = pipeline.process_csv(input_csv, output_csv, max_rows=max_rows, verbose=True)
    pipeline.print_summary()

    if args.sample:
        try:
            report = score(str(REPO_ROOT / "support_tickets" / "sample_support_tickets.csv"), output_csv)
            print_score_report(report)
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(f"Scoring failed: {e}")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())