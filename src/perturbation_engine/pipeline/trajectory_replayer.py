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
        For osworld-human-main: ["`CLICK` the text box labeled 'Search'", ...]
        For osworld-verified: [{"action": "pyautogui.click(89, 76)", ...}, ...]
        """
        if self.current_step >= len(self.trajectory_data):
            self.logger.warning(
                f"Trajectory step {self.current_step} out of range (max: {len(self.trajectory_data) - 1})"
            )
            return "", ""  # Return empty action when trajectory is complete

        step_data = self.trajectory_data[self.current_step]
        self.current_step += 1

        # Handle different trajectory formats
        if isinstance(step_data, str):
            # osworld-human-main format: direct action strings
            action = step_data
        elif isinstance(step_data, dict) and "action" in step_data:
            # osworld-verified format: JSON objects with action field
            action = step_data["action"]
        else:
            self.logger.error(f"Unexpected trajectory data format: {type(step_data)}")
            return "", ""

        return "", action

    def get_total_steps(self) -> int:
        """Get total steps in the trajectory"""
        return len(self.trajectory_data)

    def is_complete(self) -> bool:
        """Check if trajectory is complete"""
        return self.current_step >= len(self.trajectory_data)

    def get_current_step(self) -> int:
        """Get current step index"""
        return self.current_step
