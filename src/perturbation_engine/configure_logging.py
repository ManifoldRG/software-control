import json
import logging
import os
import sys
import threading
from collections import deque
from datetime import datetime
from typing import Optional


class ContextLogHandler(logging.Handler):
    """Log handler that saves context around errors and warnings to debug folder"""

    def __init__(self, debug_dir: str = "./debug", context_lines: int = 20):
        super().__init__()
        self.debug_dir = debug_dir
        self.context_lines = context_lines
        self.log_buffer = deque(maxlen=context_lines * 2)  # Store more than needed
        self.lock = threading.Lock()
        self.current_run_id = None
        self.current_trajectory_id = None

    def set_run_context(self, trajectory_id: str, run_id: str):
        """Set the current run context for organizing debug logs"""
        with self.lock:
            self.current_trajectory_id = trajectory_id
            self.current_run_id = run_id

    def emit(self, record: logging.LogRecord):
        """Emit a log record and save context for errors/warnings"""
        with self.lock:
            # Add current record to buffer
            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "pathname": record.pathname,
                "lineno": record.lineno,
                "funcName": record.funcName,
                "process": getattr(record, "processName", "unknown"),
                "thread": record.thread,
            }
            self.log_buffer.append(log_entry)

            # Save context for errors and warnings
            if record.levelno >= logging.WARNING:
                self._save_error_context(record)

    def _save_error_context(self, error_record: logging.LogRecord):
        """Save context around an error or warning"""
        if not self.current_trajectory_id or not self.current_run_id:
            return

        try:
            # Create debug directory structure
            debug_path = os.path.join(self.debug_dir, self.current_trajectory_id, self.current_run_id, "logs")
            os.makedirs(debug_path, exist_ok=True)

            # Create context data
            context_data = {
                "error_info": {
                    "timestamp": datetime.fromtimestamp(error_record.created).isoformat(),
                    "level": error_record.levelname,
                    "logger": error_record.name,
                    "message": error_record.getMessage(),
                    "pathname": error_record.pathname,
                    "lineno": error_record.lineno,
                    "funcName": error_record.funcName,
                    "process": getattr(error_record, "processName", "unknown"),
                    "thread": error_record.thread,
                    "exc_info": error_record.exc_info,
                },
                "context_before": list(self.log_buffer)[
                    -self.context_lines - 1 : -1
                ],  # Exclude current record
                "context_after": [],  # Will be populated as more logs come in
                "total_context_lines": len(self.log_buffer),
            }

            # Save context to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"error_context_{error_record.levelname.lower()}_{timestamp}.json"
            filepath = os.path.join(debug_path, filename)

            with open(filepath, "w") as f:
                json.dump(context_data, f, indent=2)

        except Exception as e:
            # Don't let logging errors break the application
            print(f"Error saving log context: {e}", file=sys.stderr)


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


# Global context handler instance
_context_handler: Optional[ContextLogHandler] = None


def get_context_handler() -> Optional[ContextLogHandler]:
    """Get the global context handler instance"""
    return _context_handler


def set_run_context(trajectory_id: str, run_id: str):
    """Set the run context for the global context handler"""
    global _context_handler
    if _context_handler:
        _context_handler.set_run_context(trajectory_id, run_id)


def configure_logging() -> None:
    """Configure root logging for the CLI.

    - PERTURB_ENGINE_LOG_LEVEL: logging level (default: INFO)
    - PERTURB_ENGINE_LOG_COLOR: 1/0 to enable/disable ANSI colors (default: 1)
    """
    global _context_handler

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

    # Console handler for normal output
    console_handler = logging.StreamHandler(stream)
    console_handler.setLevel(level)
    console_handler.setFormatter(ColorFormatter(use_color=color_enabled))

    # Context handler for debug folder logging
    debug_dir = os.getenv("PERTURB_ENGINE_DEBUG_DIR", "./debug")
    context_lines = int(os.getenv("PERTURB_ENGINE_CONTEXT_LINES", "20"))
    _context_handler = ContextLogHandler(debug_dir=debug_dir, context_lines=context_lines)
    _context_handler.setLevel(logging.WARNING)  # Only capture warnings and errors

    root.setLevel(level)
    root.addHandler(console_handler)
    root.addHandler(_context_handler)


def configure_subprocess_logging(process_name: str = None) -> None:
    """Configure logging for subprocesses to show in main process.

    This ensures subprocess logs appear in the main process console.
    """
    global _context_handler

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

    console_handler = logging.StreamHandler(stream)
    console_handler.setLevel(level)

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

    console_handler.setFormatter(formatter)

    # Context handler for debug folder logging (same as main process)
    debug_dir = os.getenv("PERTURB_ENGINE_DEBUG_DIR", "./debug")
    context_lines = int(os.getenv("PERTURB_ENGINE_CONTEXT_LINES", "20"))
    _context_handler = ContextLogHandler(debug_dir=debug_dir, context_lines=context_lines)
    _context_handler.setLevel(logging.WARNING)  # Only capture warnings and errors

    root.setLevel(level)
    root.addHandler(console_handler)
    root.addHandler(_context_handler)
