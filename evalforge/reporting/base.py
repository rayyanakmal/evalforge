"""Reporter abstract base class — contract for all output formats.

Each reporter transforms a RunResult into a specific output format
(JSON file, terminal table, diff table, etc.).

Implementations: JSONReporter, ConsoleReporter, DiffReporter
"""

from abc import ABC, abstractmethod
from pathlib import Path

from evalforge.models.result import RunResult


class Reporter(ABC):
    """Abstract base for all output formats.

    Each reporter transforms a RunResult into a specific output format.
    """

    @abstractmethod
    def generate(self, result: RunResult) -> str:
        """Convert RunResult to string representation.

        Args:
            result: The complete run result with all test results and summary.

        Returns:
            Formatted string (JSON text, table, etc.).
        """
        ...

    @abstractmethod
    def write(self, result: RunResult, path: Path) -> None:
        """Write the report to a file or stdout.

        Args:
            result: The complete run result.
            path: Destination file path (ignored by stdout-based reporters).

        Raises:
            OSError: If file cannot be written.
        """
        ...
