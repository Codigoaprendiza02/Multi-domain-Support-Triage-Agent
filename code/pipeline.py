"""End-to-end pipeline orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Generator

import pandas as pd

from agent.triage_agent import TriageAgent
from config import OUTPUT_CSV
from utils.logger import TranscriptLogger


@dataclass
class PipelineStats:
    total: int = 0
    answered: int = 0
    escalated: int = 0
    errors: int = 0


class Pipeline:
    def __init__(self, logger: TranscriptLogger | None = None) -> None:
        self.logger = logger or TranscriptLogger()
        self.agent = TriageAgent(logger=self.logger)
        self.stats = PipelineStats()

    def process_csv(
        self, input_csv: str, output_csv: str, max_rows: int | None = None, verbose: bool = True
    ) -> tuple[PipelineStats, list[dict[str, str]]]:
        """
        Process all tickets from input_csv and write results to output_csv.

        Args:
            input_csv: path to support_tickets.csv
            output_csv: path to write output.csv
            max_rows: if set, process only first N tickets (for --dry-run)
            verbose: print progress to terminal

        Returns:
            (stats, results_list)
        """
        input_path = Path(input_csv)
        if not input_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {input_csv}")

        df = pd.read_csv(input_csv)
        # Normalize column names to lowercase so 'Issue'/'issue', 'Company'/'company' both work
        df.columns = [c.lower().strip() for c in df.columns]
        if max_rows:
            df = df.head(max_rows)

        self.stats = PipelineStats(total=len(df))
        results: list[dict[str, str]] = []

        for idx, row in df.iterrows():
            try:
                ticket_id = str(row.get("ticket_id", f"TICKET_{idx}"))
                company = row.get("company")
                if pd.isna(company) or company == "" or company == "None":
                    company = None
                else:
                    company = str(company).strip()
                issue = str(row.get("issue", ""))

                result = self.agent.triage(ticket_id=ticket_id, company=company, issue=issue)
                result_dict = result.to_csv_dict()
                result_dict["ticket_id"] = ticket_id
                results.append(result_dict)

                if result.status == "answered":
                    self.stats.answered += 1
                    symbol = "✅"
                else:
                    self.stats.escalated += 1
                    symbol = "⚠️"

                product_area = result.product_area[:20] if result.product_area else "unknown"
                if verbose:
                    print(
                        f"Processing ticket [{idx + 1:03d}/{len(df):03d}]: {company or 'cross-domain':<12} | {product_area:<20} -> {result.status:<10} {symbol}"
                    )
            except Exception as exc:
                self.stats.errors += 1
                if verbose:
                    print(f"Processing ticket [{idx + 1:03d}/{len(df):03d}]: ERROR — {exc}")
                self.logger.log("ERROR", f"ticket_id={ticket_id} error={exc}")

        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df = pd.DataFrame(results)
        output_df.to_csv(output_csv, index=False)

        return self.stats, results

    def print_summary(self) -> None:
        """Print terminal summary of pipeline run."""
        pct_answered = (self.stats.answered / self.stats.total * 100) if self.stats.total > 0 else 0.0
        pct_escalated = (self.stats.escalated / self.stats.total * 100) if self.stats.total > 0 else 0.0
        print("\n" + "=" * 50)
        print("  RUN COMPLETE")
        print("-" * 50)
        print(f"  Total:      {self.stats.total}")
        print(f"  Answered:   {self.stats.answered} ({pct_answered:.1f}%)")
        print(f"  Escalated:  {self.stats.escalated} ({pct_escalated:.1f}%)")
        print(f"  Errors:     {self.stats.errors}")
        print(f"  Output:     {OUTPUT_CSV}")
        print("=" * 50 + "\n")
