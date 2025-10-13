"""
TrajectoryReplayer: Replay existing trajectories
Clean interface for trajectory replay
"""

import json
import logging
from typing import Tuple


class TrajectoryReplayer:
    """Replay existing trajectories for perturbation injection"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.trajectory_data = []
        self.current_step = 0

    def load_trajectory(self, trajectory_file_path: str):
        """Load trajectory from file"""
        try:
            with open(trajectory_file_path, "r", encoding="utf-8") as f:
                # Loads osworld-human-main trajectories
                if trajectory_file_path.startswith("osworld-human-main"):
                    json_data = json.load(f)
                    self.trajectory_data = json_data["human-ground-truth"]["single-action"]
                else:
                    # Loads osworld-verified trajectories
                    self.trajectory_data = []
                    for line in f:
                        if line.strip():
                            self.trajectory_data.append(json.loads(line))

            self.current_step = 0
            self.logger.info(f"Loaded trajectory with {len(self.trajectory_data)} steps")

        except Exception as e:
            self.logger.error(f"Error loading trajectory: {e}")
            self.trajectory_data = []

    def step(self) -> Tuple[str, str]:
        """Get next step from trajectory

        self.trajectory_data follows this format:
        [
            "`CLICK` the text box labeled 'Search'",
            "`TYPING` 'Manchester, GB'",
            "`CLICK` the entry that says 'Manchester, GB'",
            "`CLICK` the 'Monthly' tab"
        ]
        """
        action = self.trajectory_data[self.current_step]
        self.current_step += 1
        return "", action

    def get_total_steps(self) -> int:
        """Get total steps in the trajectory"""
        return len(self.trajectory_data)
