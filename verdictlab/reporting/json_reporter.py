"""JSONReporter — serializes RunResult to JSON and writes to file.

Primary persistence format consumed by compare/gate commands.
"""

import json
from pathlib import Path

from verdictlab.models.result import RunResult
from verdictlab.reporting.base import Reporter


class JSONReporter(Reporter):
    """Serialize a RunResult to JSON and write to a file.

    Usage:
        reporter = JSONReporter()
        json_str = reporter.generate(result)
        reporter.write(result, Path("verdictlab-output/report-2026.json"))
    """

    def generate(self, result: RunResult) -> str:
        """Convert RunResult to a JSON string.

        Args:
            result: The complete run result.

        Returns:
            Indented JSON string representation.
        """
        return result.model_dump_json(indent=2)

    def write(self, result: RunResult, path: Path) -> None:
        """Write the JSON report to a file.

        Creates parent directories if needed.

        Args:
            result: The complete run result.
            path: Destination file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.generate(result))
