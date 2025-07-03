"""
Console Colors and Formatting

Utilities for colored and formatted console output.
"""

import sys
from typing import Optional


class ConsoleColors:
    """Handles colored console output"""

    def __init__(self, use_colors: bool = True):
        self.use_colors = use_colors and sys.stdout.isatty()

        # ANSI color codes
        self.RESET = "\033[0m" if self.use_colors else ""
        self.BOLD = "\033[1m" if self.use_colors else ""

        # Colors
        self.RED = "\033[91m" if self.use_colors else ""
        self.GREEN = "\033[92m" if self.use_colors else ""
        self.YELLOW = "\033[93m" if self.use_colors else ""
        self.BLUE = "\033[94m" if self.use_colors else ""
        self.MAGENTA = "\033[95m" if self.use_colors else ""
        self.CYAN = "\033[96m" if self.use_colors else ""
        self.WHITE = "\033[97m" if self.use_colors else ""
        self.GRAY = "\033[90m" if self.use_colors else ""

    def _colorize(self, text: str, color: str, bold: bool = False) -> str:
        """Apply color and formatting to text"""
        if not self.use_colors:
            return text

        prefix = f"{self.BOLD if bold else ''}{color}"
        return f"{prefix}{text}{self.RESET}"

    def error(self, text: str) -> None:
        """Print error message in red"""
        print(self._colorize(text, self.RED, bold=True))

    def success(self, text: str) -> None:
        """Print success message in green"""
        print(self._colorize(text, self.GREEN, bold=True))

    def warning(self, text: str) -> None:
        """Print warning message in yellow"""
        print(self._colorize(text, self.YELLOW, bold=True))

    def info(self, text: str) -> None:
        """Print info message in blue"""
        print(self._colorize(text, self.BLUE))

    def text(self, text: str) -> None:
        """Print normal text"""
        print(text)

    def header(self, text: str) -> None:
        """Print header text in bold cyan"""
        print(self._colorize(text, self.CYAN, bold=True))

    def subheader(self, text: str) -> None:
        """Print subheader text in bold"""
        print(self._colorize(text, self.WHITE, bold=True))

    def prompt(self, text: str) -> str:
        """Format prompt text"""
        return self._colorize(text, self.CYAN)

    def muted(self, text: str) -> None:
        """Print muted text in gray"""
        print(self._colorize(text, self.GRAY))

    def table_header(self, headers: list) -> None:
        """Print table headers"""
        header_line = " | ".join(headers)
        self.subheader(header_line)
        self.muted("─" * len(header_line))

    def table_row(self, columns: list) -> None:
        """Print table row"""
        self.text(" | ".join(str(col) for col in columns))

    def progress_bar(self, current: int, total: int, width: int = 50) -> str:
        """Create a progress bar string"""
        if total == 0:
            percentage = 100
        else:
            percentage = int(current * 100 / total)

        filled = int(width * current / total) if total > 0 else 0
        bar = "█" * filled + "░" * (width - filled)

        return self._colorize(f"[{bar}] {percentage}%", self.GREEN)

    def spinner_frame(self, frame_index: int) -> str:
        """Get spinner frame for animations"""
        frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        return self._colorize(frames[frame_index % len(frames)], self.CYAN)
