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
        raw_response = step_data.get("response", {})

        # Validate action format - warn about incompatible actions
        if not self._is_valid_action(str(action)):
            self.logger.warning(
                f"Incompatible action format at step {self.current_step}: {str(action)[:100]}"
            )
            self.logger.warning(
                "Expected pyautogui actions (e.g., 'pyautogui.click(x, y)'), "
                f"got: {str(action)[:50]}. This may cause execution errors."
            )
            # Use a no-op action to prevent VM errors
            action = "pyautogui.sleep(0.1)"
            self.logger.info(f"Replaced with no-op action: {action}")

        # Handle response being either a dict or a string blob
        if isinstance(raw_response, dict):
            thought = raw_response.get("thought", "")
        else:
            thought = self._extract_thought_from_string(str(raw_response))

        response = {"thought": thought, "action": action}

        return response, [action]

    def has_more_steps(self) -> bool:
        """Check if there are more steps in the trajectory"""
        return self.current_step < len(self.trajectory_data)

    def reset(self):
        """Reset to beginning of trajectory"""
        self.current_step = 0

    def _is_valid_action(self, action: str) -> bool:
        """
        Validate that action is in a compatible format (pyautogui).

        Accepts:
        - pyautogui.click(x, y) - coordinate-based actions
        - pyautogui.hotkey('ctrl', 'c') - keyboard actions
        - import pyautogui; pyautogui.click(...) - multi-statement actions

        Rejects:
        - Custom module imports (libreoffice_calc, CalcTools, etc.)

        Args:
            action: Action string to validate

        Returns:
            True if action is valid pyautogui format, False otherwise
        """
        if not action or not isinstance(action, str):
            return False

        action_lower = action.lower().strip()

        # Check for pyautogui usage (anywhere in the action string)
        # This handles both "pyautogui.click(...)" and "import pyautogui; pyautogui.click(...)"
        if "pyautogui." in action_lower:
            # Valid pyautogui action - check it's not mixed with incompatible patterns
            incompatible_patterns = [
                "from libreoffice",  # LibreOffice custom module imports
                "import libreoffice",
                "from calc import",
                "from writer import",
                "calctools.",  # Custom tool classes (case insensitive)
                "writertools.",
            ]

            for pattern in incompatible_patterns:
                if pattern in action_lower:
                    return False

            return True

        # Check for incompatible formats that will cause errors
        incompatible_patterns = [
            "from libreoffice",
            "import libreoffice",
            "from calc import",
            "from writer import",
            "calctools.",
            "writertools.",
        ]

        for pattern in incompatible_patterns:
            if pattern in action_lower:
                return False

        # If it doesn't contain pyautogui, it might be a plain Python expression
        # Accept it but log a warning
        self.logger.debug(f"Action doesn't contain 'pyautogui.': {action[:50]}")
        return True

    def _extract_thought_from_string(self, text: str) -> str:
        """
        Extract the 'thought' content from a string response. Supports:
        - <think>...</think> blocks
        - Fallback to plain text without code fences
        """
        try:
            lower = text.lower()
            start_tag = "<think>"
            end_tag = "</think>"
            if start_tag in lower and end_tag in lower:
                # Find tag positions in original text preserving case
                s = lower.find(start_tag)
                e = lower.find(end_tag)
                if s != -1 and e != -1 and e > s:
                    inner = text[s + len(start_tag) : e]
                    return inner.strip()
            # Strip code fences if present
            if "```" in text:
                parts = text.split("```")
                # Return the non-code part preceding the first fence if meaningful
                prefix = parts[0].strip()
                if prefix:
                    return prefix
                # Otherwise, try any remaining non-empty part
                for p in parts:
                    p = p.strip()
                    if p and not p.startswith(("python", "bash", "javascript", "js")):
                        return p
            return text.strip()
        except Exception:
            return text.strip()
