"""
Autoglm_v Integration: Clean interface for autoglm_v tools
Provides app state extraction, element identification, and coordinate tracking
"""

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Import existing autoglm_v logic
from perturbation_engine.tools.autoglm_v.prompt.accessibility_tree_handle import (
    filter_nodes,
    find_active_applications,
)
from perturbation_engine.tools.autoglm_v.prompt.grounding_agent import GroundingAgent
from perturbation_engine.tools.autoglm_v.tools.package.code import CodeTools
from perturbation_engine.tools.autoglm_v.tools.package.google_chrome import BrowserTools

# from perturbation_engine.tools.autoglm_v.tools.package.libreoffice_calc import CalcTools
# from perturbation_engine.tools.autoglm_v.tools.package.libreoffice_writer import WriterTools
# from perturbation_engine.tools.autoglm_v.tools.package.libreoffice_impress import ImpressTools
from perturbation_engine.tools.autoglm_v.tools.package.vlc import VLCTools


def extract_coordinate_from_node(
    node: ET.Element, platform: str = "Ubuntu"
) -> Optional[Tuple[int, int, int, int]]:
    """Extract coordinates and size from XML node based on accessibility_tree_handle.py logic"""
    try:
        if platform == "Ubuntu":
            component_ns = "https://accessibility.ubuntu.example.org/ns/component"
        elif platform == "Windows":
            component_ns = "https://accessibility.windows.example.org/ns/component"
        else:
            return None

        # Extract coordinates and size using the same logic as accessibility_tree_handle.py
        coords_str = node.get(f"{{{component_ns}}}screencoord", "")
        size_str = node.get(f"{{{component_ns}}}size", "")

        if not coords_str or not size_str:
            return None

        # Parse coordinates: "(x, y)" format
        coords_match = re.match(r"\((\d+), (\d+)\)", coords_str)
        size_match = re.match(r"\((\d+), (\d+)\)", size_str)

        if not coords_match or not size_match:
            return None

        x, y = int(coords_match.group(1)), int(coords_match.group(2))
        w, h = int(size_match.group(1)), int(size_match.group(2))

        # Calculate center coordinates
        center_x = x + w // 2
        center_y = y + h // 2

        return (center_x, center_y, w, h)

    except Exception:
        return None


def extract_text_from_node(node: ET.Element, platform: str = "Ubuntu") -> str:
    """Extract text from XML node based on accessibility_tree_handle.py logic"""
    try:
        # Use the same logic as linearize_accessibility_tree function
        text = node.text if node.text is not None else ""
        text = text.strip()
        name = node.get("name", "").strip()

        if text == "":
            text = name
        elif name != "" and text != name:
            text = f"{name} ({text})"

        # Handle special case for Windows EditWrapper
        if platform == "Windows":
            value_ns = "https://accessibility.windows.example.org/ns/value"
            class_ns = "https://accessibility.windows.example.org/ns/class"
            if node.get(f"{{{class_ns}}}", "").endswith("EditWrapper") and node.get(f"{{{value_ns}}}", ""):
                text = node.get(f"{{{value_ns}}}", "")

        return text.replace("\n", "\\n")

    except Exception:
        return ""


@dataclass
class AppElement:
    """Represents a UI element with position and properties"""

    element_id: str
    element_type: str
    name: str
    text: str
    position: Dict[str, int]  # center_x, center_y, width, height
    properties: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert AppElement to dictionary for JSON serialization"""
        return {
            "element_id": self.element_id,
            "element_type": self.element_type,
            "name": self.name,
            "text": self.text,
            "position": self.position,
            "properties": self.properties,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppElement":
        """Create AppElement from dictionary"""
        return cls(
            element_id=data["element_id"],
            element_type=data["element_type"],
            name=data["name"],
            text=data["text"],
            position=data["position"],
            properties=data["properties"],
        )


@dataclass
class AppState:
    """Represents the state of an application"""

    app_name: str
    app_type: str
    window_title: str
    elements: List[AppElement]
    properties: Dict[str, Any]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert AppState to dictionary for JSON serialization"""
        return {
            "app_name": self.app_name,
            "app_type": self.app_type,
            "window_title": self.window_title,
            "elements": [element.to_dict() for element in self.elements],
            "properties": self.properties,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppState":
        """Create AppState from dictionary"""
        return cls(
            app_name=data["app_name"],
            app_type=data["app_type"],
            window_title=data["window_title"],
            elements=[AppElement.from_dict(elem_data) for elem_data in data["elements"]],
            properties=data["properties"],
            timestamp=data["timestamp"],
        )


class AutoglmAppStateExtractor:
    """Extract app states using autoglm_v tools"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Initialize autoglm_v tool classes
        self.tools = {
            "code": CodeTools,
            "chrome": BrowserTools,
            # 'libreoffice_calc': CalcTools,
            # 'libreoffice_writer': WriterTools,
            # 'libreoffice_impress': ImpressTools,
            "vlc": VLCTools,
        }

    def extract_app_states(self, accessibility_tree: str) -> List[AppState]:
        """Extract app states using existing accessibility_tree_handle.py logic with preserved hierarchy"""
        try:
            app_states = []

            # Parse the XML structure
            root = ET.fromstring(accessibility_tree)

            # Use existing logic to find active applications
            platform = "Ubuntu"
            if platform == "Ubuntu":
                _state_ns = "https://accessibility.ubuntu.example.org/ns/state"
            else:
                _state_ns = "https://accessibility.windows.example.org/ns/state"

            keep_apps = find_active_applications(ET.ElementTree(root), _state_ns)

            # Process each application while preserving hierarchy
            for app_element in root:
                if app_element.tag == "application":
                    app_name = app_element.get("name", "Unknown")

                    # Only process active applications (same logic as linearize_accessibility_tree)
                    if app_name in keep_apps:
                        app_state = self._extract_app_state_with_context(app_element, app_name, platform)
                        if app_state and app_state.elements:
                            app_states.append(app_state)

            self.logger.info(
                f"Extracted {len(app_states)} app states using enhanced accessibility_tree_handle logic"
            )
            return app_states

        except Exception as e:
            self.logger.error(f"Error extracting app states: {e}")
            return []

    def _extract_app_state_with_context(
        self, app_element: ET.Element, app_name: str, platform: str
    ) -> Optional[AppState]:
        """Extract app state using enhanced filtering logic to avoid invisible/duplicate elements"""
        try:
            # Use existing filter_nodes logic to get valid elements
            filtered_nodes = filter_nodes(app_element, platform, check_image=True)

            elements = []
            seen_elements = set()  # Track seen elements to avoid duplicates

            for node in filtered_nodes:
                # Extract coordinates using existing logic
                coords = extract_coordinate_from_node(node, platform)
                if not coords:
                    continue

                center_x, center_y, width, height = coords

                # Extract text using existing logic
                text = extract_text_from_node(node)

                # Enhanced filtering to avoid invisible/duplicate elements
                if not self._is_valid_interactive_element(
                    node, text, center_x, center_y, width, height, platform
                ):
                    continue

                # Create unique identifier to avoid duplicates
                element_key = f"{node.tag}:{text}:{center_x}:{center_y}:{width}:{height}"
                if element_key in seen_elements:
                    continue
                seen_elements.add(element_key)

                # Create element with preserved app context
                element = AppElement(
                    element_id=f"element_{len(elements)}",
                    element_type=node.tag,
                    name=text,
                    text=text,
                    position={"center_x": center_x, "center_y": center_y, "width": width, "height": height},
                    properties={
                        "app_context": app_name,
                        "raw_line": f"{node.tag}\t{text}\t({center_x}, {center_y})\t({width}, {height})",
                        "node_attributes": dict(node.attrib),  # Store original attributes for debugging
                    },
                )
                elements.append(element)

            if not elements:
                return None

            # Determine app type
            app_type = self._map_app_name_to_type(app_name)
            properties = self._get_app_properties(app_type)

            return AppState(
                app_name=app_name,
                app_type=app_type,
                window_title=app_name,
                elements=elements,
                properties=properties,
                timestamp=self._get_timestamp(),
            )

        except Exception as e:
            self.logger.error(f"Error extracting app state from {app_name}: {e}")
            return None

    def _is_valid_interactive_element(
        self,
        node: ET.Element,
        text: str,
        center_x: int,
        center_y: int,
        width: int,
        height: int,
        platform: str,
    ) -> bool:
        """Enhanced filtering to identify truly interactive elements and avoid invisible/duplicate ones"""
        try:
            # Basic size and position checks
            if width <= 0 or height <= 0 or center_x < 0 or center_y < 0:
                return False

            # Check for minimum interactive size (too small elements are likely invisible)
            if width < 5 or height < 5:
                return False

            # Check accessibility states for Ubuntu
            if platform == "Ubuntu":
                state_ns = "https://accessibility.ubuntu.example.org/ns/state"

                # Must be visible and showing
                if (
                    node.get(f"{{{state_ns}}}visible", "false") != "true"
                    or node.get(f"{{{state_ns}}}showing", "false") != "true"
                ):
                    return False

                # Check for enabled state (interactive elements should be enabled)
                enabled = node.get(f"{{{state_ns}}}enabled", "false") == "true"
                _editable = node.get(f"{{{state_ns}}}editable", "false") == "true"
                checkable = node.get(f"{{{state_ns}}}checkable", "false") == "true"
                expandable = node.get(f"{{{state_ns}}}expandable", "false") == "true"

                # For menu-item elements, check for actual interactivity
                if node.tag == "menu-item":
                    # Must have meaningful text and be enabled
                    if not text or len(text.strip()) == 0 or not enabled:
                        return False

                    # Check if it's actually interactive by looking for action attributes
                    # Look for action-related attributes (ShowMenu, Click, etc.)
                    has_action = any(
                        attr.startswith("act:")
                        or attr.endswith("_desc")
                        or attr.endswith("_kb")
                        or "ShowMenu" in attr
                        or "Click" in attr
                        or "Press" in attr
                        for attr in node.attrib.keys()
                    )

                    # Also check for interactive states
                    has_interactive_state = (
                        checkable or expandable or node.get(f"{{{state_ns}}}sensitive", "false") == "true"
                    )

                    if not has_action and not has_interactive_state:
                        # This might be a non-interactive menu item (just a label)
                        return False

                # For check-box elements, prioritize them over menu-item for same text
                if node.tag == "check-box" and text:
                    # Check-box elements are typically more specific and interactive
                    return True

            # Text-based filtering
            if not text or len(text.strip()) == 0:
                # Only allow elements without text if they have specific interactive properties
                logging.warning(f"Element at {center_x}, {center_y} without text: {node.tag}")
                if node.tag in ["button", "check-box", "slider", "scroll-bar"]:
                    return True
                return False

            # Avoid elements with generic or empty names
            generic_names = {"", " ", "  ", "unknown", "unnamed", "null", "none"}
            if text.lower().strip() in generic_names:
                return False

            # Avoid elements that are likely decorative or non-interactive
            decorative_patterns = [
                r"^\s*$",  # Empty or whitespace only
                r"^[0-9\s\-\.]+$",  # Only numbers, spaces, dashes, dots
                r"^[^\w\s]+$",  # Only special characters
            ]

            import re

            for pattern in decorative_patterns:
                if re.match(pattern, text.strip()):
                    return False

            return True

        except Exception as e:
            self.logger.debug(f"Error in element validation: {e}")
            return True  # Default to allowing the element if validation fails

    def _parse_linearized_tree(self, linearized_tree: str) -> List[AppState]:
        """Parse linearized accessibility tree to extract app states with proper hierarchy"""
        try:
            app_states = []
            lines = linearized_tree.strip().split("\n")

            if len(lines) < 2:  # Need at least header + 1 data line
                return []

            # Skip header line
            data_lines = lines[1:]

            # Group elements by application using spatial clustering and context
            app_elements = {}

            for line in data_lines:
                if not line.strip():
                    continue

                # Parse line format: "tag\ttext\tposition (center x & y)\tsize (w & h)"
                parts = line.split("\t")
                if len(parts) < 4:
                    continue

                tag, text, position_str, size_str = parts

                # Extract coordinates and size
                try:
                    # Position format: "(x, y)"
                    pos_match = re.match(r"\((\d+), (\d+)\)", position_str)
                    if not pos_match:
                        continue
                    center_x, center_y = int(pos_match.group(1)), int(pos_match.group(2))

                    # Size format: "(w, h)"
                    size_match = re.match(r"\((\d+), (\d+)\)", size_str)
                    if not size_match:
                        continue
                    width, height = int(size_match.group(1)), int(size_match.group(2))

                    # Determine app using improved logic
                    app_name = self._determine_app_from_element(tag, text, center_x, center_y, width, height)

                    if app_name not in app_elements:
                        app_elements[app_name] = []

                    # Create AppElement
                    element = AppElement(
                        element_id=f"element_{len(app_elements[app_name])}",
                        element_type=tag,
                        name=text,
                        text=text,
                        position={
                            "center_x": center_x,
                            "center_y": center_y,
                            "width": width,
                            "height": height,
                        },
                        properties={"raw_line": line},
                    )
                    app_elements[app_name].append(element)

                except (ValueError, IndexError) as e:
                    self.logger.debug(f"Error parsing line: {line}, error: {e}")
                    continue

            # Create AppState objects
            for app_name, elements in app_elements.items():
                if elements:  # Only create states for apps with elements
                    app_type = self._map_app_name_to_type(app_name)
                    properties = self._get_app_properties(app_type)

                    app_state = AppState(
                        app_name=app_name,
                        app_type=app_type,
                        window_title=app_name,
                        elements=elements,
                        properties=properties,
                        timestamp=self._get_timestamp(),
                    )
                    app_states.append(app_state)

            return app_states

        except Exception as e:
            self.logger.error(f"Error parsing linearized tree: {e}")
            return []

    def _map_app_name_to_type(self, app_name: str) -> str:
        """Map application name to autoglm_v tool type"""
        app_name_lower = app_name.lower()

        if "code" in app_name_lower or "vscode" in app_name_lower:
            return "code"
        elif "chrome" in app_name_lower or "browser" in app_name_lower:
            return "chrome"
        elif "calc" in app_name_lower or "spreadsheet" in app_name_lower:
            return "libreoffice_calc"
        elif "writer" in app_name_lower or "document" in app_name_lower:
            return "libreoffice_writer"
        elif "impress" in app_name_lower or "presentation" in app_name_lower:
            return "libreoffice_impress"
        elif "vlc" in app_name_lower or "media" in app_name_lower:
            return "vlc"
        else:
            return "unknown"

    def _get_app_properties(self, app_type: str) -> Dict[str, Any]:
        """Get application-specific properties using autoglm_v tools"""
        try:
            if app_type in self.tools:
                tool_class = self.tools[app_type]

                # Get environment info from the tool
                if hasattr(tool_class, "env_info"):
                    result = tool_class.env_info()
                    return {"env_info": result}

                # Get specific properties based on app type
                if app_type == "libreoffice_calc":
                    if hasattr(tool_class, "get_workbook_info"):
                        result = tool_class.get_workbook_info()
                        return {"workbook_info": result}

                elif app_type == "chrome":
                    return {"browser_state": "active"}

                elif app_type == "code":
                    return {"editor_state": "active"}

            return {}

        except Exception as e:
            self.logger.debug(f"Error getting app properties for {app_type}: {e}")
            return {}

    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        import datetime

        return datetime.datetime.now().strftime("%Y%m%d@%H%M%S")


class AutoglmElementTracker:
    """Track UI elements using autoglm_v tools"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.grounding_agent = GroundingAgent()
        from perturbation_engine.pipeline.clean_llm_services import CleanElementIdentificationLLM

        self.llm = CleanElementIdentificationLLM()

    def identify_target_element(self, action_str: str, app_states: List[AppState]) -> Optional[AppElement]:
        """
        Identify target element using LLM-based approach only.

        This method uses an LLM to intelligently identify the target element from the action string
        and available app states. The LLM returns element identifiers, and we find the actual
        element with coordinates from the app states.
        """
        try:
            # Convert AppState objects to dictionaries for LLM processing
            app_states_dict = [
                app_state.to_dict() if hasattr(app_state, "to_dict") else app_state
                for app_state in app_states
            ]

            # Use LLM to identify target element
            llm_result = self._identify_with_llm(action_str, app_states_dict)
            if llm_result:
                # Find the actual element in app states using LLM's identifier
                target_element = self._find_element_by_identifier(llm_result, app_states)
                if target_element:
                    self.logger.info(
                        f"✓ LLM identified target element: {target_element.name} ({target_element.element_type})"
                    )
                    return target_element

            self.logger.warning(f"✗ Could not identify target element for: {action_str[:100]}")
            return None

        except Exception as e:
            self.logger.error(f"Error identifying target element: {e}")
            return None

    def _identify_with_llm(
        self, action_str: str, app_states: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Use LLM to identify target element"""
        try:
            retries = 0
            while retries < 3:
                retries += 1
                result = self.llm.identify_target_element(action_str, app_states)
                if result:
                    return result
                else:
                    self.logger.warning(f"LLM failed to identify target element. Retrying ({retries}/3)...")
            return None

        except Exception as e:
            self.logger.error(f"Error with LLM element identification: {e}")
            return None

    def _find_element_by_identifier(
        self, llm_result: Dict[str, Any], app_states: List[AppState]
    ) -> Optional[AppElement]:
        """Find the actual element in app states using LLM's identifier"""
        try:
            # Extract identifiers from LLM result
            target_name = llm_result.get("name", "").lower()
            target_type = llm_result.get("element_type", "").lower()
            target_app = llm_result.get("app_name", "").lower()

            if not target_name:
                return None

            # Search through all app states
            for app_state in app_states:
                # Check if this is the right app
                if target_app and target_app not in app_state.app_name.lower():
                    continue

                # Search elements in this app
                for element in app_state.elements:
                    element_name_lower = element.name.lower()
                    element_type_lower = element.element_type.lower()

                    # Match by name (exact or partial)
                    name_match = (
                        element_name_lower == target_name
                        or target_name in element_name_lower
                        or element_name_lower in target_name
                    )

                    # Match by type if specified
                    type_match = (
                        not target_type
                        or element_type_lower == target_type
                        or target_type in element_type_lower
                    )

                    if name_match and type_match:
                        # Add LLM metadata to the element
                        element.properties.update(
                            {
                                "llm_identified": True,
                                "llm_confidence": llm_result.get("confidence", 1.0),
                                "llm_reasoning": llm_result.get("reasoning", ""),
                                "llm_app_context": target_app,
                            }
                        )
                        return element

            return None

        except Exception as e:
            self.logger.error(f"Error finding element by identifier: {e}")
            return None

    def track_element_after_perturbation(
        self, target_element: AppElement, app_states: List[AppState]
    ) -> Optional[AppElement]:
        """Track element after perturbation to see if it moved"""
        try:
            # Find element with same properties in new app states
            for app_state in app_states:
                for element in app_state.elements:
                    if (
                        element.element_type == target_element.element_type
                        and element.name == target_element.name
                        and element.text == target_element.text
                    ):
                        return element

            return None

        except Exception as e:
            self.logger.error(f"Error tracking element after perturbation: {e}")
            return None

    def update_action_coordinates(self, action_str: str, new_position: Dict[str, int]) -> Tuple[str, bool]:
        """Update action coordinates to the exact center of the new position."""

        new_x = new_position["center_x"]
        new_y = new_position["center_y"]

        coord_pattern = r"(\d+),\s*(\d+)"

        def replace_coords(match):
            return f"{new_x}, {new_y}"

        updated_action = re.sub(coord_pattern, replace_coords, action_str)
        return updated_action


class AutoglmPerturbationGenerator:
    """Enhanced perturbation generator with comprehensive GUI randomization"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.grounding_agent = GroundingAgent()
        self.tools = {
            "code": CodeTools,
            "chrome": BrowserTools,
            "google_chrome": BrowserTools,
            # 'libreoffice_calc': CalcTools,
            # 'libreoffice_writer': WriterTools,
            # 'libreoffice_impress': ImpressTools,
            "vlc": VLCTools,
        }

        # Comprehensive perturbation strategies for each app type
        self.perturbation_strategies = {
            "chrome": {
                "theme": ["dark_mode", "light_mode", "high_contrast", "custom_colors"],
                "layout": ["compact_mode", "expanded_mode", "sidebar_position", "tab_arrangement"],
                "content_variation": ["bookmark_changes", "history_modification", "extension_injection"],
                "ui_injection": ["notification_popup", "extension_banner", "update_prompt"],
                "window_management": ["window_resize", "tab_rearrangement", "focus_changes"],
            },
            "code": {
                "theme": ["dark_theme", "light_theme", "high_contrast", "custom_colors"],
                "layout": ["sidebar_position", "panel_arrangement", "editor_layout", "terminal_position"],
                "content_variation": ["file_tree_changes", "extension_modification", "settings_changes"],
                "ui_injection": ["extension_notification", "update_prompt", "error_popup"],
                "window_management": ["window_resize", "panel_toggle", "focus_changes"],
            },
            "vlc": {
                "theme": ["dark_theme", "light_theme", "custom_colors"],
                "layout": ["control_position", "playlist_layout", "window_arrangement"],
                "content_variation": ["playlist_changes", "media_info_modification"],
                "ui_injection": ["notification_popup", "error_dialog", "update_prompt"],
                "window_management": ["window_resize", "fullscreen_toggle", "control_visibility"],
            },
            "libreoffice_calc": {
                "theme": ["dark_theme", "light_theme", "custom_colors"],
                "layout": ["toolbar_position", "sheet_arrangement", "panel_layout"],
                "content_variation": ["data_modification", "formatting_changes", "sheet_reorganization"],
                "ui_injection": ["notification_popup", "error_dialog", "save_prompt"],
                "window_management": ["window_resize", "panel_toggle", "sheet_navigation"],
            },
        }

    def generate_perturbation_command(
        self, target_app: str, perturbation_type: str, parameters: Dict[str, Any]
    ) -> str:
        """Generate comprehensive perturbation command with realistic GUI randomization"""
        try:
            app_type = target_app.lower()

            # Get perturbation strategy for this app
            strategy = self.perturbation_strategies.get(app_type, {})
            available_perturbations = strategy.get(perturbation_type, [])

            if not available_perturbations:
                # Fallback to generic perturbation
                return self._generate_generic_perturbation(perturbation_type, parameters)

            # Select random perturbation from available options
            import random

            selected_perturbation = random.choice(available_perturbations)

            # Generate app-specific perturbation
            if app_type in ["chrome", "google_chrome"]:
                return self._generate_chrome_perturbation(
                    selected_perturbation, perturbation_type, parameters
                )
            elif app_type == "code":
                return self._generate_code_perturbation(selected_perturbation, perturbation_type, parameters)
            elif app_type == "vlc":
                return self._generate_vlc_perturbation(selected_perturbation, perturbation_type, parameters)
            elif app_type in ["libreoffice_calc", "libreoffice_writer", "libreoffice_impress"]:
                return self._generate_libreoffice_perturbation(
                    selected_perturbation, perturbation_type, parameters
                )

            # Fallback to generic perturbation
            return self._generate_generic_perturbation(perturbation_type, parameters)

        except Exception as e:
            self.logger.error(f"Error generating perturbation command: {e}")
            return ""

    def _generate_chrome_perturbation(
        self, selected_perturbation: str, perturbation_type: str, parameters: Dict[str, Any]
    ) -> str:
        """Generate Chrome-specific perturbation with realistic GUI randomization"""
        try:
            if selected_perturbation == "dark_mode":
                code = "BrowserTools.open_appearance_settings()"
                return self.grounding_agent.tool_commands(code, "google_chrome")[0]
            elif selected_perturbation == "light_mode":
                code = "BrowserTools.open_appearance_settings()"
                return self.grounding_agent.tool_commands(code, "google_chrome")[0]
            elif selected_perturbation == "bookmark_changes":
                code = "BrowserTools.bookmark_page()"
                return self.grounding_agent.tool_commands(code, "google_chrome")[0]
            elif selected_perturbation == "extension_injection":
                code = "BrowserTools.open_extensions()"
                return self.grounding_agent.tool_commands(code, "google_chrome")[0]
            elif selected_perturbation == "notification_popup":
                code = "BrowserTools.open_privacy_settings()"
                return self.grounding_agent.tool_commands(code, "google_chrome")[0]
            else:
                # Default Chrome perturbation
                code = "BrowserTools.open_appearance_settings()"
                return self.grounding_agent.tool_commands(code, "google_chrome")[0]
        except Exception as e:
            self.logger.error(f"Error generating Chrome perturbation: {e}")
            return "BrowserTools.open_appearance_settings()"

    def _generate_code_perturbation(
        self, selected_perturbation: str, perturbation_type: str, parameters: Dict[str, Any]
    ) -> str:
        """Generate VS Code-specific perturbation with realistic GUI randomization"""
        try:
            if selected_perturbation == "dark_theme":
                code = "CodeTools.install_extension('ms-vscode.theme-materialdark')"
                return self.grounding_agent.tool_commands(code, "code")[0]
            elif selected_perturbation == "light_theme":
                code = "CodeTools.install_extension('ms-vscode.theme-lightplus')"
                return self.grounding_agent.tool_commands(code, "code")[0]
            elif selected_perturbation == "extension_notification":
                code = "CodeTools.install_extension('ms-python.python')"
                return self.grounding_agent.tool_commands(code, "code")[0]
            elif selected_perturbation == "settings_changes":
                code = "CodeTools.toggle_sync('on')"
                return self.grounding_agent.tool_commands(code, "code")[0]
            elif selected_perturbation == "file_tree_changes":
                code = "CodeTools.add_folder('/tmp/test_folder')"
                return self.grounding_agent.tool_commands(code, "code")[0]
            else:
                # Default VS Code perturbation
                code = "CodeTools.install_extension('ms-vscode.theme-materialdark')"
                return self.grounding_agent.tool_commands(code, "code")[0]
        except Exception as e:
            self.logger.error(f"Error generating VS Code perturbation: {e}")
            return "CodeTools.install_extension('ms-vscode.theme-materialdark')"

    def _generate_vlc_perturbation(
        self, selected_perturbation: str, perturbation_type: str, parameters: Dict[str, Any]
    ) -> str:
        """Generate VLC-specific perturbation with realistic GUI randomization"""
        try:
            if selected_perturbation == "dark_theme":
                code = "VLCTools.set_settings('qt-theme', 'dark')"
                return self.grounding_agent.tool_commands(code, "vlc")[0]
            elif selected_perturbation == "light_theme":
                code = "VLCTools.set_settings('qt-theme', 'light')"
                return self.grounding_agent.tool_commands(code, "vlc")[0]
            elif selected_perturbation == "playlist_changes":
                code = "VLCTools.add_to_playlist('file:///tmp/test.mp3')"
                return self.grounding_agent.tool_commands(code, "vlc")[0]
            elif selected_perturbation == "notification_popup":
                code = "VLCTools.set_settings('qt-notification', '1')"
                return self.grounding_agent.tool_commands(code, "vlc")[0]
            elif selected_perturbation == "window_resize":
                code = "VLCTools.toggle_fullscreen(True)"
                return self.grounding_agent.tool_commands(code, "vlc")[0]
            else:
                # Default VLC perturbation
                code = "VLCTools.pause()"
                return self.grounding_agent.tool_commands(code, "vlc")[0]
        except Exception as e:
            self.logger.error(f"Error generating VLC perturbation: {e}")
            return "VLCTools.pause()"

    def _generate_libreoffice_perturbation(
        self, selected_perturbation: str, perturbation_type: str, parameters: Dict[str, Any]
    ) -> str:
        """Generate LibreOffice-specific perturbation with realistic GUI randomization"""
        try:
            if selected_perturbation == "dark_theme":
                # LibreOffice theme changes via system settings
                return "gsettings set org.gnome.desktop.interface gtk-theme 'Adwaita-dark'"
            elif selected_perturbation == "light_theme":
                return "gsettings set org.gnome.desktop.interface gtk-theme 'Adwaita'"
            elif selected_perturbation == "data_modification":
                # Generic data modification for LibreOffice
                return "echo 'LibreOffice data modification applied'"
            elif selected_perturbation == "notification_popup":
                return "notify-send 'LibreOffice' 'Document modified'"
            elif selected_perturbation == "window_resize":
                return "wmctrl -r 'LibreOffice' -e 0,100,100,800,600"
            else:
                # Default LibreOffice perturbation
                return "gsettings set org.gnome.desktop.interface gtk-theme 'Adwaita-dark'"
        except Exception as e:
            self.logger.error(f"Error generating LibreOffice perturbation: {e}")
            return "gsettings set org.gnome.desktop.interface gtk-theme 'Adwaita-dark'"

    def _generate_generic_perturbation(self, perturbation_type: str, parameters: Dict[str, Any]) -> str:
        """Generate generic perturbation"""
        if perturbation_type == "theme":
            return "gsettings set org.gnome.desktop.interface gtk-theme 'Adwaita-dark'"
        elif perturbation_type == "notification":
            return "notify-send 'Perturbation Applied' 'System state changed'"
        else:
            return "echo 'Generic perturbation applied'"


class AutoglmCurriculumGenerator:
    """Enhanced curriculum generator with comprehensive GUI perturbation strategies"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.extractor = AutoglmAppStateExtractor()
        self.perturbation_generator = AutoglmPerturbationGenerator()

        # Comprehensive app-specific perturbation strategies
        self.app_perturbation_strategies = {
            "chrome": {
                "theme": {
                    "realistic_scenarios": [
                        "User switches to dark mode for better visibility",
                        "System automatically applies high contrast theme",
                        "Extension changes browser appearance",
                    ],
                    "maintains_functionality": True,
                    "learning_objectives": "Adapt to different visual themes while maintaining navigation",
                },
                "layout": {
                    "realistic_scenarios": [
                        "User rearranges bookmarks toolbar",
                        "Extension modifies tab layout",
                        "System changes window arrangement",
                    ],
                    "maintains_functionality": True,
                    "learning_objectives": "Navigate UI with different layouts and arrangements",
                },
                "content_variation": {
                    "realistic_scenarios": [
                        "Bookmarks are reorganized by user",
                        "Extensions add new content",
                        "System updates change default content",
                    ],
                    "maintains_functionality": True,
                    "learning_objectives": "Handle dynamic content changes",
                },
            },
            "code": {
                "theme": {
                    "realistic_scenarios": [
                        "Developer switches to dark theme for coding",
                        "Team standardizes on specific theme",
                        "Extension automatically applies theme",
                    ],
                    "maintains_functionality": True,
                    "learning_objectives": "Work with different editor themes",
                },
                "layout": {
                    "realistic_scenarios": [
                        "Developer customizes panel layout",
                        "Extension modifies sidebar position",
                        "System changes window arrangement",
                    ],
                    "maintains_functionality": True,
                    "learning_objectives": "Navigate editor with different layouts",
                },
                "content_variation": {
                    "realistic_scenarios": [
                        "Extensions modify file tree",
                        "Settings change default behavior",
                        "Updates modify interface elements",
                    ],
                    "maintains_functionality": True,
                    "learning_objectives": "Adapt to dynamic editor content",
                },
            },
            "vlc": {
                "theme": {
                    "realistic_scenarios": [
                        "User switches to dark theme for media viewing",
                        "System applies accessibility theme",
                        "Plugin changes media player appearance",
                    ],
                    "maintains_functionality": True,
                    "learning_objectives": "Control media player with different themes",
                },
                "layout": {
                    "realistic_scenarios": [
                        "User rearranges control panel",
                        "Plugin modifies playlist layout",
                        "System changes window arrangement",
                    ],
                    "maintains_functionality": True,
                    "learning_objectives": "Navigate player with different layouts",
                },
                "content_variation": {
                    "realistic_scenarios": [
                        "Playlist is modified by user",
                        "Plugin adds new media sources",
                        "System updates change default content",
                    ],
                    "maintains_functionality": True,
                    "learning_objectives": "Handle dynamic media content",
                },
            },
        }

    def generate_scenario_specs(
        self, seed_trajectory, app_states: List[AppState], curriculum_config
    ) -> List[Dict[str, Any]]:
        """Generate comprehensive scenario specifications with realistic GUI perturbations"""
        try:
            scenarios = []

            # Analyze task complexity and app states
            task_analysis = self._analyze_task_for_perturbation(seed_trajectory.task_instruction)

            # Generate scenarios for each detected application
            for app_state in app_states:
                if app_state.app_type != "unknown":
                    app_scenarios = self._create_app_specific_scenarios(
                        app_state, seed_trajectory, task_analysis
                    )
                    scenarios.extend(app_scenarios)

            # Generate cross-app scenarios
            cross_app_scenarios = self._create_cross_app_scenarios(app_states, seed_trajectory, task_analysis)
            scenarios.extend(cross_app_scenarios)

            # Ensure we have enough scenarios
            while len(scenarios) < curriculum_config.scenario_count:
                generic_scenario = self._create_generic_scenario(seed_trajectory, task_analysis)
                scenarios.append(generic_scenario)

            # Prioritize scenarios based on task relevance
            prioritized_scenarios = self._prioritize_scenarios(scenarios, task_analysis)

            return prioritized_scenarios[: curriculum_config.scenario_count]

        except Exception as e:
            self.logger.error(f"Error generating scenario specs: {e}")
            return []

    def _analyze_task_for_perturbation(self, task_instruction: str) -> Dict[str, Any]:
        """Analyze task to determine appropriate perturbation strategies"""
        # Simple keyword-based analysis (could be enhanced with LLM)
        task_lower = task_instruction.lower()

        analysis = {
            "complexity": "moderate",
            "domain": "general",
            "perturbation_sensitivity": "medium",
            "critical_elements": [],
            "recommended_perturbations": [],
        }

        # Determine domain
        if any(word in task_lower for word in ["browser", "chrome", "web", "url", "bookmark"]):
            analysis["domain"] = "web"
            analysis["recommended_perturbations"].extend(["theme", "layout", "content_variation"])
        elif any(word in task_lower for word in ["code", "editor", "file", "programming"]):
            analysis["domain"] = "development"
            analysis["recommended_perturbations"].extend(["theme", "layout", "content_variation"])
        elif any(word in task_lower for word in ["media", "video", "audio", "vlc", "play"]):
            analysis["domain"] = "multimedia"
            analysis["recommended_perturbations"].extend(["theme", "layout", "content_variation"])
        elif any(word in task_lower for word in ["office", "document", "calc", "writer"]):
            analysis["domain"] = "office"
            analysis["recommended_perturbations"].extend(["theme", "layout", "content_variation"])

        # Determine complexity
        if len(task_instruction.split()) > 20:
            analysis["complexity"] = "complex"
            analysis["perturbation_sensitivity"] = "high"
        elif len(task_instruction.split()) > 10:
            analysis["complexity"] = "moderate"
        else:
            analysis["complexity"] = "simple"
            analysis["perturbation_sensitivity"] = "low"

        return analysis

    def _create_app_specific_scenarios(
        self, app_state: AppState, seed_trajectory, task_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create scenarios specific to the application"""
        scenarios = []
        app_strategies = self.app_perturbation_strategies.get(app_state.app_type, {})

        for perturbation_type, strategy_info in app_strategies.items():
            # Select realistic scenario
            import random

            realistic_scenario = random.choice(strategy_info["realistic_scenarios"])

            # Generate perturbation command
            perturbation_command = self.perturbation_generator.generate_perturbation_command(
                app_state.app_type, perturbation_type, {}
            )

            scenario = {
                "scenario_id": f"scenario_{app_state.app_name}_{perturbation_type}_{hash(realistic_scenario)}",
                "target_app": app_state.app_name,
                "perturbation_trigger": f"When interacting with {app_state.app_name}",
                "available_perturbation_actions": perturbation_command,
                "learning_objectives": strategy_info["learning_objectives"],
                "target_components": [elem.element_type for elem in app_state.elements[:3]],
                "perturbation_types": [perturbation_type],
                "realistic_scenario": realistic_scenario,
                "maintains_functionality": strategy_info["maintains_functionality"],
                "perturbation_intensity": self._determine_intensity(perturbation_type, task_analysis),
            }
            scenarios.append(scenario)

        return scenarios

    def _create_cross_app_scenarios(
        self, app_states: List[AppState], seed_trajectory, task_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create scenarios that involve multiple applications"""
        scenarios = []

        if len(app_states) > 1:
            # Create window management scenario
            scenario = {
                "scenario_id": f"cross_app_window_management_{hash(seed_trajectory.task_instruction)}",
                "target_app": "system",
                "perturbation_trigger": "During multi-app task execution",
                "available_perturbation_actions": "wmctrl -r :ACTIVE: -e 0,100,100,800,600",
                "learning_objectives": "Handle window management changes across applications",
                "target_components": ["window", "desktop"],
                "perturbation_types": ["window_management"],
                "realistic_scenario": "System automatically rearranges windows for better workflow",
                "maintains_functionality": True,
                "perturbation_intensity": "medium",
            }
            scenarios.append(scenario)

            # Create theme consistency scenario
            scenario = {
                "scenario_id": f"cross_app_theme_consistency_{hash(seed_trajectory.task_instruction)}",
                "target_app": "system",
                "perturbation_trigger": "When switching between applications",
                "available_perturbation_actions": "gsettings set org.gnome.desktop.interface gtk-theme 'Adwaita-dark'",
                "learning_objectives": "Maintain task flow across applications with different themes",
                "target_components": ["theme", "interface"],
                "perturbation_types": ["theme"],
                "realistic_scenario": "System applies consistent theme across all applications",
                "maintains_functionality": True,
                "perturbation_intensity": "low",
            }
            scenarios.append(scenario)

        return scenarios

    def _create_generic_scenario(self, seed_trajectory, task_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create generic scenario when app-specific scenarios are insufficient"""
        return {
            "scenario_id": f"generic_scenario_{hash(seed_trajectory.task_instruction)}",
            "target_app": "system",
            "perturbation_trigger": "During task execution",
            "available_perturbation_actions": "notify-send 'System Update' 'Background process running'",
            "learning_objectives": "Learn to handle system-level perturbations",
            "target_components": ["desktop", "notification"],
            "perturbation_types": ["notification"],
            "realistic_scenario": "System notification appears during task execution",
            "maintains_functionality": True,
            "perturbation_intensity": "low",
        }

    def _determine_intensity(self, perturbation_type: str, task_analysis: Dict[str, Any]) -> str:
        """Determine perturbation intensity based on type and task sensitivity"""
        if task_analysis["perturbation_sensitivity"] == "high":
            return "low"
        elif task_analysis["perturbation_sensitivity"] == "low":
            return "high"
        else:
            # Medium sensitivity - vary by perturbation type
            if perturbation_type in ["theme", "notification"]:
                return "low"
            elif perturbation_type in ["layout", "content_variation"]:
                return "medium"
            else:
                return "high"

    def _prioritize_scenarios(
        self, scenarios: List[Dict[str, Any]], task_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Prioritize scenarios based on task relevance and learning objectives"""
        # Sort by relevance to task domain
        domain_priority = {
            "web": ["chrome", "browser"],
            "development": ["code", "editor"],
            "multimedia": ["vlc", "media"],
            "office": ["libreoffice", "calc", "writer"],
        }

        def scenario_priority(scenario):
            target_app = scenario["target_app"].lower()
            domain = task_analysis["domain"]

            # Higher priority for scenarios matching task domain
            if domain in domain_priority:
                if any(app in target_app for app in domain_priority[domain]):
                    return 1

            # Medium priority for system-level scenarios
            if scenario["target_app"] == "system":
                return 2

            # Lower priority for other scenarios
            return 3

        return sorted(scenarios, key=scenario_priority)


if __name__ == "__main__":
    # Test the LLM-based element identification
    print("Autoglm_v Integration - LLM-based element identification ready")
