"""
Simplified App State Extractor - Uses comprehensive environment state
Extracts app state directly from PerturbationDesktopEnv._get_obs()
"""

import logging
from dataclasses import dataclass
from typing import List


@dataclass
class AppState:
    """Minimal app state for curriculum generation"""

    app_type: str
    current_view: str  # "spreadsheet", "document", "browser", "image_editor"
    key_elements: list[str]  # ["button", "input", "menu"]
    task_context: str  # Brief description of current task


class SimpleAppStateExtractor:
    """Simplified app state extraction using comprehensive environment data"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def extract_app_state(self, env, task_type: str, task_instruction: str) -> AppState:
        """Extract app state using comprehensive environment observation"""

        # Map task types to app categories
        app_mapping = {
            "chrome": "browser",
            "libreoffice_calc": "spreadsheet",
            "libreoffice_writer": "document",
            "libreoffice_impress": "presentation",
            "gimp": "image_editor",
            "vs_code": "code_editor",
            "os": "file_manager",
            "thunderbird": "email_client",
            "vlc": "media_player",
        }

        app_type = app_mapping.get(task_type, "unknown")

        # Get comprehensive environment observation
        try:
            obs = env._get_obs()
        except Exception as e:
            self.logger.warning(f"Could not get environment observation: {e}")
            obs = {}

        # Extract state using comprehensive observation data
        return self._extract_state_from_obs(obs, app_type, task_instruction)

    def _extract_state_from_obs(self, obs: dict, app_type: str, task_instruction: str) -> AppState:
        """Extract state from comprehensive observation"""

        # Determine current view from environment data
        current_view = self._determine_current_view(obs, app_type, task_instruction)

        # Extract key elements based on app type and DOM structure
        key_elements = self._extract_key_elements(obs, app_type)

        return AppState(
            app_type=app_type,
            current_view=current_view,
            key_elements=key_elements,
            task_context=task_instruction,
        )

    def _determine_current_view(self, obs: dict, app_type: str, task_instruction: str) -> str:
        """Determine current view from environment data"""

        if app_type == "browser":
            # Use URL and page title to determine view
            url = obs.get("url", "").lower()
            title = obs.get("page_title", "").lower()

            if (
                "spreadsheet" in task_instruction.lower()
                or "calc" in url
                or "sheets" in url
                or "spreadsheet" in title
            ):
                return "spreadsheet"
            elif "document" in task_instruction.lower() or "docs" in url or "document" in title:
                return "document"
            elif "presentation" in task_instruction.lower() or "slides" in url or "presentation" in title:
                return "presentation"
            else:
                return "webpage"

        elif app_type == "spreadsheet":
            return "spreadsheet"
        elif app_type == "document":
            return "document"
        elif app_type == "presentation":
            return "presentation"
        elif app_type == "image_editor":
            return "image_editor"
        elif app_type == "code_editor":
            return "code_editor"
        elif app_type == "file_manager":
            return "file_manager"
        elif app_type == "email_client":
            return "email_client"
        elif app_type == "media_player":
            return "media_player"
        else:
            return "unknown"

    def _extract_key_elements(self, obs: dict, app_type: str) -> List[str]:
        """Extract key UI elements based on app type"""

        # Base elements for all apps
        base_elements = ["button", "input", "menu"]

        # App-specific elements
        if app_type == "browser":
            return base_elements + ["link", "form", "div", "span"]
        elif app_type == "spreadsheet":
            return base_elements + ["cell", "formula", "table", "range"]
        elif app_type == "document":
            return base_elements + ["text", "paragraph", "toolbar"]
        elif app_type == "presentation":
            return base_elements + ["slide", "text", "image", "canvas"]
        elif app_type == "image_editor":
            return base_elements + ["canvas", "tool", "palette", "brush"]
        elif app_type == "code_editor":
            return base_elements + ["editor", "file", "terminal"]
        elif app_type == "file_manager":
            return base_elements + ["file", "folder", "path"]
        elif app_type == "email_client":
            return base_elements + ["email", "list", "folder"]
        elif app_type == "media_player":
            return base_elements + ["player", "slider", "playlist"]
        else:
            return base_elements
