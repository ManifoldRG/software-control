"""Trajectory replayer for replaying existing task trajectories with perturbations"""

import json
import logging
import os
from typing import Any, Dict, List, Tuple


class TrajectoryReplayer:
    """Replays existing task trajectories step by step"""

    def __init__(self, trajectory_file_path: str):
        """Initialize trajectory replayer

        Args:
            trajectory_file_path: Path to the trajectory directory or file
        """
        self.trajectory_file_path = trajectory_file_path
        self.trajectory_steps = []
        self.current_step = 0
        self.logger = logging.getLogger(__name__)

        # Load trajectory data
        self._load_trajectory()

    def _load_trajectory(self):
        """Load trajectory steps from trajectory file"""
        try:
            # Check if it's a directory with traj.jsonl
            if os.path.isdir(self.trajectory_file_path):
                traj_file = os.path.join(self.trajectory_file_path, "traj.jsonl")
            else:
                traj_file = self.trajectory_file_path

            if not os.path.exists(traj_file):
                self.logger.warning(f"Trajectory file not found: {traj_file}")
                return

            with open(traj_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            step_data = json.loads(line)
                            self.trajectory_steps.append(step_data)
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Failed to parse trajectory step: {e}")
                            continue

            self.logger.info(f"Loaded {len(self.trajectory_steps)} trajectory steps from {traj_file}")

        except Exception as e:
            self.logger.error(f"Error loading trajectory: {e}")
            self.trajectory_steps = []

    def step(self) -> Tuple[str, List[Any]]:
        """Get next step from trajectory

        Returns:
            Tuple of (response, actions) for the current step
        """
        if self.current_step >= len(self.trajectory_steps):
            # No more steps, return empty response
            return "", []

        step_data = self.trajectory_steps[self.current_step]
        self.current_step += 1

        # Extract response and action from step data
        response = step_data.get("response", "")
        action = step_data.get("action")

        # Convert single action to list if needed
        actions = [action] if action is not None else []

        return response, actions

    def reset(self):
        """Reset to beginning of trajectory"""
        self.current_step = 0

    def has_more_steps(self) -> bool:
        """Check if there are more steps in the trajectory"""
        return self.current_step < len(self.trajectory_steps)

    def get_total_steps(self) -> int:
        """Get total number of steps in trajectory"""
        return len(self.trajectory_steps)

    def get_current_step_info(self) -> Dict[str, Any]:
        """Get information about current step"""
        if self.current_step < len(self.trajectory_steps):
            return self.trajectory_steps[self.current_step]
        return {}
