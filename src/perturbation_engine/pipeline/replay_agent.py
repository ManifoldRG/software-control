import logging


class ReplayAgent:
    """Agent that replays the trajectory"""

    def __init__(self, trajectory_folder_dir: str):
        self.logger = logging.getLogger(__name__)
        self.trajectory = self._load_trajectory(trajectory_folder_dir)

    def _load_trajectory(self, trajectory_folder_dir: str):
        """Load the trajectory from the trajectory folder"""
        pass

    def step(self):
        """Get the next action"""
        pass

    def reset(self, trajectory_folder_dir: str):
        """Reset the agent"""
        self.trajectory = self._load_trajectory(trajectory_folder_dir)
