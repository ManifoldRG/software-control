"""
PerturbationDesktopEnv: Extended env with chrome management
Clean interface for environment management
"""

import logging
import os
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Tuple

from OSWorld.desktop_env.desktop_env import DesktopEnv
from perturbation_engine.control.perturbation_controller import PerturbationController


class AppType(Enum):
    """Application types"""

    BROWSER = "browser"
    LIBREOFFICE_CALC = "libreoffice_calc"
    LIBREOFFICE_WRITER = "libreoffice_writer"
    LIBREOFFICE_IMPRESS = "libreoffice_impress"
    VS_CODE = "vs_code"
    GIMP = "gimp"
    VLC = "vlc"
    THUNDERBIRD = "thunderbird"
    FILE_MANAGER = "file_manager"
    TERMINAL = "terminal"
    UNKNOWN = "unknown"


class PerturbationDesktopEnv(DesktopEnv):
    """Enhanced DesktopEnv that provides perturbation controller"""

    def __init__(
        self,
        provider_name: str = "vmware",
        region: str = None,
        path_to_vm: str = None,
        snapshot_name: str = "init_state",
        action_space: str = "pyautogui",
        cache_dir: str = "cache",
        screen_size: Tuple[int] = (
            int(os.environ.get("SCREEN_WIDTH", 1920)),
            int(os.environ.get("SCREEN_HEIGHT", 1080)),
        ),
        headless: bool = False,
        require_a11y_tree: bool = True,
        require_terminal: bool = False,
        os_type: str = "Ubuntu",
        enable_proxy: bool = False,
        client_password: str = "",
        chromium_port: int = 9222,
    ):
        self.logger = logging.getLogger(__name__)
        self.chromium_port = chromium_port

        super().__init__(
            provider_name=provider_name,
            region=region,
            path_to_vm=path_to_vm,
            snapshot_name=snapshot_name,
            action_space=action_space,
            cache_dir=cache_dir,
            screen_size=screen_size,
            headless=headless,
            require_a11y_tree=require_a11y_tree,
            require_terminal=require_terminal,
            os_type=os_type,
            enable_proxy=enable_proxy,
            client_password=client_password,
        )

        self.logger.info("Perturbation controller initialized")

    def _start_emulator(self):
        """Override to use PerturbationController instead of PythonController"""
        super()._start_emulator()

        # Replace the controller with our enhanced version
        self.controller = PerturbationController(
            vm_ip=self.vm_ip, server_port=self.server_port, chromium_port=self.chromium_port
        )
        self.logger.info("Replaced controller with PerturbationController")

    def close(self) -> None:
        """Close both the perturbation controller and original environment"""
        if hasattr(self.controller, "close_playwright"):
            self.controller.close_playwright()
        super().close()

    def get_obs(self):
        """Get comprehensive observation including DOM, A11Y, and app-specific state"""
        try:
            return {
                "screenshot": self.controller.get_screenshot(),
                "accessibility_tree": self.controller.get_accessibility_tree()
                if self.require_a11y_tree
                else None,
                "terminal": self.controller.get_terminal_output() if self.require_terminal else None,
                "app_states": self.get_app_states_from_accessibility_tree(),
                "instruction": self.instruction,
                "timestamp": self._get_timestamp(),
                "url": getattr(self.controller, "current_url", ""),
                "window_size": getattr(self.controller, "window_size", {}),
            }

        except Exception as e:
            self.logger.error(f"Error getting observation: {e}")
            return {
                "screenshot": None,
                "accessibility_tree": None,
                "terminal": None,
                "app_states": [],
                "instruction": self.instruction,
                "timestamp": self._get_timestamp(),
                "url": "",
                "window_size": {},
            }

    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        import datetime

        return datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

    def get_app_states_from_accessibility_tree(self) -> List[Dict[str, Any]]:
        """Extract simplified app states from accessibility tree XML for LLM prompting"""
        try:
            a11y_tree = self.controller.get_accessibility_tree()
            if a11y_tree is None:
                self.logger.warning("Accessibility tree is None")
                return []

            # Parse XML and extract app states
            import xml.etree.ElementTree as ET

            try:
                root = ET.fromstring(a11y_tree)
            except ET.ParseError as e:
                self.logger.warning(f"Failed to parse accessibility tree XML: {e}")
                return []

            # Group elements by application
            app_groups = self._group_elements_by_application(root)
            app_states = []

            for app_name, elements in app_groups.items():
                if not elements:
                    continue

                app_type = self._detect_app_type_from_name(app_name)
                if app_type == "unknown":
                    continue
                # Create app state for this application
                app_state = {
                    "app_type": app_type,
                    "current_view": self._detect_current_view(elements),
                    "key_elements": self._extract_key_elements_for_app(elements),
                    "task_context": f"Application: {app_name}",
                    "element_count": len(elements),
                    "app_name": app_name,
                }

                app_states.append(app_state)

            # If no apps found, create a generic state
            if not app_states:
                app_states.append(
                    {
                        "app_type": "unknown",
                        "current_view": "unknown",
                        "key_elements": [],
                        "task_context": "No accessible applications detected",
                        "element_count": 0,
                        "app_name": "unknown",
                    }
                )

            self.logger.info(f"Extracted {len(app_states)} app states from accessibility tree")
            return app_states

        except Exception as e:
            self.logger.error(f"Error extracting app info from accessibility tree: {e}")
            return []

    def _group_elements_by_application(self, root) -> Dict[str, List[Dict[str, Any]]]:
        """Group accessibility elements by application"""
        app_groups = defaultdict(list)

        try:
            # Create parent map for tree traversal
            parent_map = {child: parent for parent in root.iter() for child in parent}

            # Find all elements and group by application
            for elem in root.iter():
                # Get application name from the element or its parent
                app_name = self._get_application_name(elem, parent_map)
                if not app_name or app_name == "unknown":
                    continue

                # Extract element information
                element_info = {
                    "role": elem.get("role", ""),
                    "name": elem.get("name", ""),
                    "description": elem.get("description", ""),
                    "value": elem.get("value", ""),
                    "tag": elem.tag,
                    "attributes": dict(elem.attrib),
                }

                app_groups[app_name].append(element_info)

            return dict(app_groups)

        except Exception as e:
            self.logger.warning(f"Error grouping elements by application: {e}")
            return {}

    def _get_application_name(self, elem, parent_map: Dict) -> str:
        """Extract application name from element or its parents using parent map"""
        try:
            current_elem = elem
            visited = set()  # Prevent infinite loops

            while current_elem is not None and current_elem not in visited:
                visited.add(current_elem)

                # Check current element for application info
                app_name = current_elem.get("application", "")
                if app_name:
                    return app_name

                # Check for window or frame elements that might contain app info
                if current_elem.tag in ["window", "frame", "application"]:
                    name = current_elem.get("name", "")
                    if name:
                        return name

                # Move to parent element
                current_elem = parent_map.get(current_elem)

            return "unknown"

        except Exception as e:
            self.logger.warning(f"Error getting application name: {e}")
            return "unknown"

    def _detect_app_type_from_name(self, app_name: str) -> str:
        """Detect application type from application name"""
        app_name_lower = app_name.lower()

        if any(browser in app_name_lower for browser in ["chrome", "firefox", "safari", "edge", "browser"]):
            return "browser"
        elif any(office in app_name_lower for office in ["libreoffice", "calc", "writer", "impress"]):
            if "calc" in app_name_lower:
                return "libreoffice_calc"
            elif "writer" in app_name_lower:
                return "libreoffice_writer"
            elif "impress" in app_name_lower:
                return "libreoffice_impress"
            else:
                return "libreoffice"
        elif any(code in app_name_lower for code in ["code", "vscode", "editor"]):
            return "vs_code"
        elif "gimp" in app_name_lower:
            return "gimp"
        elif "vlc" in app_name_lower:
            return "vlc"
        elif "thunderbird" in app_name_lower:
            return "thunderbird"
        elif any(file in app_name_lower for file in ["file", "manager", "explorer"]):
            return "file_manager"
        elif any(term in app_name_lower for term in ["terminal", "bash", "shell"]):
            return "terminal"
        else:
            return "unknown"

    def _detect_current_view(self, elements: List[Dict[str, Any]]) -> str:
        """Detect current view based on element types"""
        roles = [elem.get("role", "") for elem in elements]

        if "dialog" in roles:
            return "dialog_view"
        elif "menu" in roles:
            return "menu_view"
        elif "textbox" in roles and "button" in roles:
            return "form_view"
        elif "link" in roles:
            return "navigation_view"
        elif "heading" in roles:
            return "content_view"
        else:
            return "main_view"

    def _extract_key_elements_for_app(self, elements: List[Dict[str, Any]]) -> List[str]:
        """Extract key elements for a specific application"""
        key_elements = []

        # Prioritize interactive elements
        interactive_roles = ["button", "textbox", "link", "menu", "dialog"]

        for elem in elements:
            role = elem.get("role", "")
            if role in interactive_roles:
                name = elem.get("name", "")
                description = elem.get("description", "")

                element_desc = f"{role}"
                if name:
                    element_desc += f": {name}"
                if description:
                    element_desc += f" ({description})"

                key_elements.append(element_desc)

                # Limit to prevent overwhelming the LLM
                if len(key_elements) >= 10:
                    break

        return key_elements
