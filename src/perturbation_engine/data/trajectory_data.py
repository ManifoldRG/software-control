"""Trajectory data structures for the perturbation pipeline."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class MHTMLData:
    """Raw MHTML file data."""

    file_path: Path
    content: str


@dataclass
class Observation:
    """Single-step observation for an episode."""

    mhtml: MHTMLData
    image: bytes | None = None  # Screenshot of the page
    task_instruction: str | None = None


@dataclass
class Action:
    """Action (both ground truth and agent actions)."""

    action_type: str
    coordinates: tuple[int, int] | None = None
    text: str | None = None


@dataclass
class Step:
    """A single step within an episode."""

    index: int
    observation: Observation
    action: Action | None = None
    is_terminal: bool = False


@dataclass
class Episode:
    """Episodic container for structured datasets."""

    episode_id: str
    task: str
    steps: list[Step]
