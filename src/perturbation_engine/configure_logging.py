import logging
import os
import sys


class ColorFormatter(logging.Formatter):
    """A minimal ANSI color formatter for readable CLI logs.

    Colors only the level name. Falls back to plain text when disabled.
    """

    RESET = "\x1b[0m"
    COLORS = {
        "DEBUG": "\x1b[90m",  # bright black / gray
        "INFO": "\x1b[36m",  # cyan
        "WARNING": "\x1b[33m",  # yellow
        "ERROR": "\x1b[31m",  # red
        "CRITICAL": "\x1b[1;31m",  # bold red
    }

    def __init__(self, *, use_color: bool, datefmt: str | None = "%H:%M:%S") -> None:
        # Short clickable path via relative path when possible
        fmt = "%(asctime)s %(levelname)s [%(processName)s] %(pathline)s - %(message)s"
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        # Compute short clickable path: relative to CWD when available
        pathname = record.pathname
        try:
            cwd = os.getcwd()
            if pathname.startswith(cwd + os.sep):
                pathname = os.path.relpath(pathname, cwd)
        except Exception:
            pass
        record.pathline = f"{pathname}:{record.lineno}"

        if not self.use_color:
            return super().format(record)

        original_levelname = record.levelname
        color = self.COLORS.get(original_levelname)
        if color:
            record.levelname = f"{color}{original_levelname}{self.RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = original_levelname


def configure_logging() -> None:
    """Configure root logging for the CLI.

    - PERTURB_ENGINE_LOG_LEVEL: logging level (default: INFO)
    - PERTURB_ENGINE_LOG_COLOR: 1/0 to enable/disable ANSI colors (default: 1)
    """
    level_name = os.getenv("PERTURB_ENGINE_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        for handler in root.handlers:
            handler.setLevel(level)
        return

    stream = sys.stdout
    color_env = os.getenv("PERTURB_ENGINE_LOG_COLOR", "1").lower()
    color_enabled = color_env not in {"0", "false", "no"} and hasattr(stream, "isatty") and stream.isatty()

    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    handler.setFormatter(ColorFormatter(use_color=color_enabled))

    root.setLevel(level)
    root.addHandler(handler)


def configure_subprocess_logging(process_name: str = None) -> None:
    """Configure logging for subprocesses to show in main process.

    This ensures subprocess logs appear in the main process console.
    """
    level_name = os.getenv("PERTURB_ENGINE_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()

    # Clear existing handlers to avoid duplication
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # Set up console handler that writes to stdout
    stream = sys.stdout
    color_env = os.getenv("PERTURB_ENGINE_LOG_COLOR", "1").lower()
    color_enabled = color_env not in {"0", "false", "no"} and hasattr(stream, "isatty") and stream.isatty()

    handler = logging.StreamHandler(stream)
    handler.setLevel(level)

    # Create formatter with process name
    if process_name:
        formatter = ColorFormatter(use_color=color_enabled, datefmt="%H:%M:%S")
        # Override the format method to include process name
        original_format = formatter.format

        def format_with_process_name(record):
            record.processName = f"{record.processName}-{process_name}"
            return original_format(record)

        formatter.format = format_with_process_name
    else:
        formatter = ColorFormatter(use_color=color_enabled, datefmt="%H:%M:%S")

    handler.setFormatter(formatter)

    root.setLevel(level)
    root.addHandler(handler)
