"""
TrajectoryReplayer: Replay existing trajectories
Clean interface for trajectory replay
"""

import json
import logging
from typing import Any, Dict, List, Tuple


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
                self.trajectory_data = []
                for line in f:
                    if line.strip():
                        self.trajectory_data.append(json.loads(line))

            self.current_step = 0
            self.logger.info(f"Loaded trajectory with {len(self.trajectory_data)} steps")

        except Exception as e:
            self.logger.error(f"Error loading trajectory: {e}")
            self.trajectory_data = []

    def step(self) -> Tuple[Dict[str, Any], List[str]]:
        """Get next step from trajectory"""
        if self.current_step >= len(self.trajectory_data):
            return {}, []

        step_data = self.trajectory_data[self.current_step]
        self.current_step += 1

        # Extract action and response
        action = step_data.get("action", "")
        response = {"thought": step_data.get("response", {}).get("thought", ""), "action": action}

        return response, [action]

    def has_more_steps(self) -> bool:
        """Check if there are more steps in the trajectory"""
        return self.current_step < len(self.trajectory_data)

    def reset(self):
        """Reset to beginning of trajectory"""
        self.current_step = 0
