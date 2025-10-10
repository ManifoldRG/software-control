"""
Autoglm_v Integration: Clean interface for autoglm_v tools
Provides app state extraction, element identification, and coordinate tracking
"""

import logging
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

from perturbation_engine.pipeline.data_models import UIElement, VisibilityState, WindowState

# Import existing autoglm_v logic
from perturbation_engine.tools.autoglm_v.prompt.grounding_agent import GroundingAgent

# Import LibreOffice tools (with fallback if not available)
try:
    from perturbation_engine.tools.autoglm_v.tools.package.libreoffice_calc import CalcTools
    from perturbation_engine.tools.autoglm_v.tools.package.libreoffice_impress import ImpressTools
    from perturbation_engine.tools.autoglm_v.tools.package.libreoffice_writer import WriterTools

    LIBREOFFICE_TOOLS_AVAILABLE = True
except ImportError:
    # Fallback classes if LibreOffice tools are not available
    class CalcTools:
        @classmethod
        def env_info(cls):
            return "LibreOffice Calc tools not available"

    class WriterTools:
        @classmethod
        def env_info(cls):
            return "LibreOffice Writer tools not available"

    class ImpressTools:
        @classmethod
        def env_info(cls):
            return "LibreOffice Impress tools not available"

    LIBREOFFICE_TOOLS_AVAILABLE = False


# App-specific enhancement integrated directly into parsing


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


class AutoglmAppStateExtractor:
    """Enhanced app state extractor with X11, CDP, and UNO integration"""

    def __init__(self, controller=None):
        self.logger = logging.getLogger(__name__)
        self.controller = controller
        self.parser = HierarchicalParser(controller=controller)
        if not controller:
            raise ValueError("AutoglmAppStateExtractor requires a controller")

    def extract_window_states(self, accessibility_tree: str) -> List[WindowState]:
        """Extract window states with integrated app-specific enhancement"""
        try:
            # Step 1: Get real window z-order from X11
            z_order_list = self.controller.get_window_z_order()
            focused_window = self.controller.get_focused_window()
            current_desktop = self.controller.get_current_desktop()

            if z_order_list:
                self.logger.info(f"Found {len(z_order_list)} windows in X11 stacking order")
                self.logger.info(f"Focused window: {focused_window}")
            else:
                self.logger.info("No X11 windows found")

            # Step 2: Parse AT-SPI2 tree with integrated app-specific enhancement
            root = ET.fromstring(accessibility_tree)

            # Step 3: Extract windows from AT-SPI2 with integrated enhancement
            atspi_windows = self._extract_atspi_windows(root)

            if not atspi_windows:
                self.logger.warning("No windows found in AT-SPI2 tree")
                return []

            # Step 4: Match with X11 windows and assign real z-order
            if not z_order_list:
                self.logger.warning("No X11 windows found")
                raise
            window_states = self._match_x11_with_atspi(
                atspi_windows, z_order_list, focused_window, current_desktop
            )

            self.logger.info(f"Extracted {len(window_states)} window states")
            return window_states

        except Exception as e:
            self.logger.error(f"Error extracting window states: {e}")
            return []

    def _extract_atspi_windows(self, root: ET.Element) -> List[Dict[str, Any]]:
        """Extract windows from AT-SPI2 tree"""
        windows = []

        # Debug: Log all applications found
        all_apps = []
        for app_node in root:
            if app_node.tag == "application":
                app_name = app_node.get("name", "Unknown")
                all_apps.append(app_name)

        self.logger.info(f"Found {len(all_apps)} applications in AT-SPI2 tree: {all_apps}")

        for app_node in root:
            if app_node.tag != "application":
                continue

            app_name = app_node.get("name", "Unknown")
            self.logger.debug(f"Processing application: {app_name}")

            # Skip system apps
            if self._should_skip_app(app_name):
                self.logger.debug(f"Skipping system app: {app_name}")
                continue

            # Debug: Log all node types found in this app
            node_types = set()
            for child_node in app_node:
                node_types.add(child_node.tag)
            self.logger.info(f"  Node types in {app_name}: {sorted(node_types)}")

            # Find all possible window/dialog types in this app
            # AT-SPI window types include: frame, window, dialog, popup, alert,
            # internal-frame, layered-pane, option-pane, panel, embedded, application
            window_types = {
                "frame",
                "window",
                "dialog",
                "popup",
                "alert",
                "internal-frame",
                "layered-pane",
                "option-pane",
                "panel",
                "embedded",
                "application",
            }

            # Additional heuristic: look for nodes with window-like attributes
            def is_window_like(node):
                # Check for window-like attributes
                state_ns = "https://accessibility.ubuntu.example.org/ns/state"
                has_window_attrs = (
                    node.get(f"{{{state_ns}}}modal") == "true"
                    or node.get(f"{{{state_ns}}}active") == "true"
                    or node.get("role") in ["dialog", "window", "frame", "popup"]
                )

                # Check for geometry attributes (windows typically have position/size)
                component_ns = "https://accessibility.ubuntu.example.org/ns/component"
                has_geometry = node.get(f"{{{component_ns}}}screencoord") and node.get(
                    f"{{{component_ns}}}size"
                )

                return has_window_attrs or has_geometry

            window_count = 0
            for window_node in app_node:
                # Primary check: known window types
                if window_node.tag in window_types:
                    pass  # Continue processing
                # Fallback: heuristic check for window-like nodes
                elif is_window_like(window_node):
                    self.logger.info(
                        f"  Found window-like node (heuristic): {window_node.tag} - {window_node.get('name', 'Unnamed')}"
                    )
                else:
                    continue

                window_count += 1
                window_name = window_node.get("name", "Unnamed Window")
                self.logger.info(f"  Found {window_node.tag}: {window_name}")

                # Parse elements in this window with app-specific enhancement
                elements = self.parser._parse_element_tree(
                    window_node,
                    parent_id=None,
                    depth=0,
                    parent_visibility=VisibilityState.VISIBLE,
                    app_name=app_name,
                )

                # Get all visible elements
                visible_elements = []
                if elements:
                    visible_elements = self._flatten_elements(elements)

                # Check if modal and active
                state_ns = "https://accessibility.ubuntu.example.org/ns/state"
                is_modal = window_node.get(f"{{{state_ns}}}modal") == "true"
                is_active = window_node.get(f"{{{state_ns}}}active") == "true"

                # Extract geometry from window node
                geometry = self.parser._parse_position(window_node) or {}

                windows.append(
                    {
                        "app_name": app_name,
                        "window_name": window_name,
                        "elements": visible_elements,
                        "is_modal": is_modal,
                        "is_active": is_active,
                        "geometry": geometry,
                        "atspi_node": window_node,
                    }
                )

            self.logger.info(f"App '{app_name}' contributed {window_count} windows")

        self.logger.info(f"Total windows extracted: {len(windows)}")
        return windows

    def _match_x11_with_atspi(
        self,
        atspi_windows: List[Dict[str, Any]],
        z_order_list: List[str],
        focused_window: Optional[str],
        current_desktop: int,
    ) -> List[WindowState]:
        """Match AT-SPI2 windows with X11 windows and assign real z-order"""
        window_states = []
        desktop_window_state = None

        # First pass: Find and create desktop window state
        for atspi_window in atspi_windows:
            desktop_apps = ["gnome-shell", "gjs", "desktop", "Desktop"]
            if any(desktop in atspi_window["app_name"].lower() for desktop in desktop_apps):
                self.logger.info(f"Creating desktop window state for {atspi_window['window_name']}")

                # Create desktop WindowState as the root container
                desktop_window_state = WindowState(
                    window_id="desktop_root",
                    window_name="Desktop",
                    app_name="gnome-shell",
                    is_active=False,  # Desktop is the background container
                    is_modal=False,
                    geometry={"x": 0, "y": 0, "width": 1920, "height": 1080},  # Full screen
                    z_order=0,  # Will be adjusted based on visible windows
                    x11_window_id=None,  # Desktop doesn't have traditional X11 window
                    is_mapped=True,
                    desktop=current_desktop,
                    root_element=self._build_element_tree(atspi_window["elements"]),
                )
                break

        # Second pass: Process all other windows
        for atspi_window in atspi_windows:
            desktop_apps = ["gnome-shell", "gjs", "desktop", "Desktop"]
            if any(desktop in atspi_window["app_name"].lower() for desktop in desktop_apps):
                # Skip desktop in this pass - already processed above
                continue

            # Find matching X11 window
            x11_window_id = self._find_matching_x11_window(
                atspi_window["app_name"], atspi_window["window_name"]
            )

            if not x11_window_id:
                self.logger.debug(f"No X11 window found for {atspi_window['window_name']}")
                continue

            # Get real geometry from X11
            geometry = self.controller.get_window_geometry(x11_window_id)
            if not geometry:
                continue

            # Get z-order position
            try:
                z_order = len(z_order_list) - z_order_list.index(x11_window_id)
            except ValueError:
                z_order = 0

            # Get desktop
            window_desktop = self.controller.get_window_desktop(x11_window_id)

            # Create WindowState with real data
            window_state = WindowState(
                window_id=f"{atspi_window['app_name']}_{atspi_window['window_name']}",
                window_name=atspi_window["window_name"],
                app_name=atspi_window["app_name"],
                is_active=(x11_window_id == focused_window),
                is_modal=atspi_window["is_modal"],
                geometry=geometry,
                z_order=z_order,
                x11_window_id=x11_window_id,
                is_mapped=geometry.get("mapped", True),
                desktop=window_desktop,
                root_element=self._build_element_tree(atspi_window["elements"]),
            )

            # Include all windows on current desktop (including minimized/hidden)
            if window_desktop == -1 or window_desktop == current_desktop:
                window_states.append(window_state)

        # Calculate desktop z-order: should be higher than minimized/hidden windows
        if desktop_window_state and window_states:
            # Find the lowest z-order among visible windows
            visible_windows = [w for w in window_states if w.is_mapped]
            if visible_windows:
                min_visible_z_order = min(w.z_order for w in visible_windows)
                # Desktop should be just below the lowest visible window
                desktop_window_state.z_order = min_visible_z_order - 1
            else:
                # If no visible windows, desktop is at z-order 0
                desktop_window_state.z_order = 0

            self.logger.info(f"Desktop z-order set to {desktop_window_state.z_order}")

        # Add desktop to window states
        if desktop_window_state:
            window_states.append(desktop_window_state)
            self.logger.info("Desktop window state added")

        # Sort by z-order (highest first = topmost)
        window_states.sort(key=lambda w: w.z_order, reverse=True)

        self.logger.info(f"Matched {len(window_states)} windows with desktop as root")
        for w in window_states:
            self.logger.info(
                f"  z={w.z_order}: {w.window_name} ({w.app_name}) - {len(w.root_element.children) if w.root_element else 0} elements"
            )

        return window_states

    def _find_matching_x11_window(self, app_name: str, window_name: str) -> Optional[str]:
        """Find X11 window ID that matches AT-SPI2 window"""

        # Special handling for desktop environment
        desktop_apps = ["gnome-shell", "gjs", "desktop", "Desktop"]
        if any(desktop in app_name.lower() for desktop in desktop_apps):
            # For desktop, try to find the root window or a special desktop window
            try:
                # Try to find desktop-related windows
                desktop_window_ids = self.controller.find_windows_for_app("gnome-shell")
                if desktop_window_ids:
                    # Use the first desktop window found
                    return desktop_window_ids[0]

                # Fallback: try to find any window with "desktop" in the name
                for window_id in self.controller.get_window_z_order():
                    x11_title = self.controller.get_window_name(window_id)
                    if "desktop" in x11_title.lower() or "gnome" in x11_title.lower():
                        return window_id

            except Exception as e:
                self.logger.debug(f"Could not find X11 window for desktop: {e}")

        # Get candidate windows from X11
        window_ids = self.controller.find_windows_for_app(app_name)

        # Match by window title
        for window_id in window_ids:
            x11_title = self.controller.get_window_name(window_id)

            # Exact match
            if window_name == x11_title:
                return window_id

            # Partial match (for cases like "file.txt - VS Code" vs "Visual Studio Code")
            if window_name in x11_title or x11_title in window_name:
                return window_id

        # If only one window, assume it's the one
        if len(window_ids) == 1:
            return window_ids[0]

        return None

    def _flatten_elements(self, root_element: UIElement) -> List[UIElement]:
        """Flatten element tree to list"""
        elements = [root_element]

        def traverse(elem: UIElement):
            for child in elem.children:
                elements.append(child)
                traverse(child)

        traverse(root_element)
        return elements

    def _build_element_tree(self, elements: List[UIElement]) -> Optional[UIElement]:
        """Build element tree from flat list with deduplication"""
        if not elements:
            return None

        # Deduplicate elements by element_id (keep first occurrence)
        seen_ids = set()
        unique_elements = []
        for elem in elements:
            if elem.element_id not in seen_ids:
                seen_ids.add(elem.element_id)
                unique_elements.append(elem)

        # Find root element (no parent)
        root_elements = [elem for elem in unique_elements if elem.parent_id is None]
        if not root_elements:
            return unique_elements[0]  # Fallback to first element

        root = root_elements[0]

        # Clear all children first to avoid duplicates
        for elem in unique_elements:
            elem.children.clear()

        # Build parent-child relationships
        element_map = {elem.element_id: elem for elem in unique_elements}

        for elem in unique_elements:
            if elem.parent_id and elem.parent_id in element_map:
                parent = element_map[elem.parent_id]
                parent.children.append(elem)

        return root

    def _should_skip_app(self, app_name: str) -> bool:
        """Skip system/background apps but include desktop environment"""
        skip_apps = ["vmware-user", "gsd-", "ibus-", "evolution-alarm", "xdg-desktop-portal"]

        # Always include desktop environment apps
        desktop_apps = ["gnome-shell", "gjs", "desktop", "Desktop"]
        if any(desktop in app_name.lower() for desktop in desktop_apps):
            return False

        return any(skip in app_name for skip in skip_apps)


class HierarchicalParser:
    """Parses accessibility tree into hierarchical structure with integrated app-specific enhancement"""

    def __init__(self, platform: str = "Ubuntu", controller=None):
        self.platform = platform
        self.controller = controller
        self.state_ns = "https://accessibility.ubuntu.example.org/ns/state"
        self.component_ns = "https://accessibility.ubuntu.example.org/ns/component"
        self.element_counter = 0

    def parse_window(self, window_node: ET.Element, app_name: str) -> WindowState:
        """Parse a window and its element tree"""
        window_state = WindowState(
            window_id=f"{app_name}_{window_node.get('name', 'window')}",
            window_name=window_node.get("name", "Unknown Window"),
            app_name=app_name,
            is_active=window_node.get(f"{{{self.state_ns}}}active") == "true",
            is_modal=window_node.get(f"{{{self.state_ns}}}modal") == "true",
            geometry=self._parse_geometry(window_node),
            z_order=self._estimate_z_order(window_node),
        )

        # Parse element tree recursively
        window_state.root_element = self._parse_element_tree(
            window_node, parent_id=None, depth=0, parent_visibility=VisibilityState.VISIBLE
        )

        return window_state

    def _parse_element_tree(
        self,
        node: ET.Element,
        parent_id: Optional[str],
        depth: int,
        parent_visibility: VisibilityState,
        app_name: str = "",
    ) -> Optional[UIElement]:
        """Recursively parse element and its children with integrated app-specific enhancement"""

        # Determine this element's visibility
        visibility = self._determine_visibility(node, parent_visibility)

        # Parse position (may be None for structural elements)
        position = self._parse_position(node)
        if not position and not self._is_structural_element(node):
            # Skip elements without position unless they're containers
            return None

        # Create element
        element = UIElement(
            element_id=self._generate_id(),
            element_type=node.tag,
            name=self._extract_name(node),
            position=position or {},
            parent_id=parent_id,
            depth=depth,
            visibility=visibility,
            is_enabled=node.get(f"{{{self.state_ns}}}enabled") == "true",
            is_focused=node.get(f"{{{self.state_ns}}}focused") == "true",
            is_expanded=self._is_expanded(node),
            properties=self._extract_properties(node),
        )

        # Parse children - but skip if parent is hidden
        if visibility not in [
            VisibilityState.HIDDEN_COLLAPSED,
            VisibilityState.HIDDEN_WINDOW,
            VisibilityState.HIDDEN_TAB,
            VisibilityState.HIDDEN_NOT_SHOWING,
        ]:
            for child_node in node:
                child_element = self._parse_element_tree(
                    child_node,
                    parent_id=element.element_id,
                    depth=depth + 1,
                    parent_visibility=visibility,
                    app_name=app_name,
                )
                if child_element:
                    element.children.append(child_element)

        # Integrate app-specific enhancement directly during parsing
        if self.controller and app_name:
            self._enhance_element_with_app_data(element, app_name)

        return element

    def _determine_visibility(self, node: ET.Element, parent_visibility: VisibilityState) -> VisibilityState:
        """Determine if element is truly visible"""

        # Inherit parent's hidden state
        if parent_visibility in [
            VisibilityState.HIDDEN_COLLAPSED,
            VisibilityState.HIDDEN_WINDOW,
            VisibilityState.HIDDEN_TAB,
            VisibilityState.HIDDEN_NOT_SHOWING,
        ]:
            return parent_visibility

        # Check AT-SPI2 states
        showing = node.get(f"{{{self.state_ns}}}showing") == "true"
        visible = node.get(f"{{{self.state_ns}}}visible") == "true"

        if not showing or not visible:
            return VisibilityState.HIDDEN_NOT_SHOWING

        # Check if it's inside a collapsed container
        if self._is_collapsed_container(node):
            return VisibilityState.HIDDEN_COLLAPSED

        # Structural elements are visible but marked as such
        if self._is_structural_element(node):
            return VisibilityState.STRUCTURAL

        return VisibilityState.VISIBLE

    def _is_collapsed_container(self, node: ET.Element) -> bool:
        """Check if this is a collapsed menu/dropdown"""
        if node.tag in ["popup-menu", "menu"]:
            # Popup menus without showing state are collapsed
            showing = node.get(f"{{{self.state_ns}}}showing")
            return showing != "true"

        if node.tag in ["combo-box"]:
            expanded = node.get(f"{{{self.state_ns}}}expanded")
            return expanded != "true"

        return False

    def _is_expanded(self, node: ET.Element) -> bool:
        """Check if expandable element is expanded"""
        if node.tag in ["menu-item", "combo-box", "tree-item"]:
            return node.get(f"{{{self.state_ns}}}expanded") == "true"
        return False

    def _is_structural_element(self, node: ET.Element) -> bool:
        """Check if element is a structural container"""
        return node.tag in [
            "frame",
            "panel",
            "filler",
            "layered-pane",
            "unknown",
            "status-bar",
            "menu-bar",
            "tool-bar",
        ]

    def _parse_position(self, node: ET.Element) -> Optional[Dict[str, int]]:
        """Extract position and size"""
        try:
            coords_str = node.get(f"{{{self.component_ns}}}screencoord", "")
            size_str = node.get(f"{{{self.component_ns}}}size", "")

            if not coords_str or not size_str:
                return None

            # Parse "(x, y)" format
            coords = eval(coords_str)
            size = eval(size_str)

            if coords[0] < 0 or coords[1] < 0 or size[0] <= 0 or size[1] <= 0:
                return None

            return {
                "x": coords[0],
                "y": coords[1],
                "width": size[0],
                "height": size[1],
                "center_x": coords[0] + size[0] // 2,
                "center_y": coords[1] + size[1] // 2,
            }
        except Exception:
            return None

    def _parse_geometry(self, node: ET.Element) -> Dict[str, int]:
        """Parse window geometry"""
        pos = self._parse_position(node)
        return pos if pos else {}

    def _extract_name(self, node: ET.Element) -> str:
        """Extract element name/text"""
        name = node.get("name", "").strip()
        text = (node.text or "").strip()

        if not name:
            return text
        if not text or name == text:
            return name
        return f"{name} ({text})"

    def _extract_properties(self, node: ET.Element) -> Dict[str, Any]:
        """Extract additional properties"""
        props = {}

        # Value for sliders, etc.
        val_ns = "https://accessibility.ubuntu.example.org/ns/value"
        if node.get(f"{{{val_ns}}}value"):
            props["value"] = node.get(f"{{{val_ns}}}value")
            props["min"] = node.get(f"{{{val_ns}}}min")
            props["max"] = node.get(f"{{{val_ns}}}max")

        # Actions
        act_ns = "https://accessibility.ubuntu.example.org/ns/action"
        actions = []
        for attr, _value in node.attrib.items():
            if attr.startswith(f"{{{act_ns}}}") and attr.endswith("_desc"):
                action_name = attr.replace(f"{{{act_ns}}}", "").replace("_desc", "")
                actions.append(action_name)
        if actions:
            props["actions"] = actions

        # Image flag
        if node.get("image") == "true":
            props["is_image"] = True

        return props

    def _estimate_z_order(self, window_node: ET.Element) -> int:
        """Estimate window z-order"""
        z_order = 0

        if window_node.get(f"{{{self.state_ns}}}active") == "true":
            z_order += 1000
        if window_node.get(f"{{{self.state_ns}}}modal") == "true":
            z_order += 500
        if window_node.get("role") in ["dialog", "popup"]:
            z_order += 300

        return z_order

    def _generate_id(self) -> str:
        """Generate unique element ID"""
        self.element_counter += 1
        return f"elem_{self.element_counter}"

    def _enhance_element_with_app_data(self, element: UIElement, app_name: str):
        """Enhance element with app-specific data using controller methods"""
        try:
            app_type = self._map_app_name_to_type(app_name)

            if app_type == "chrome" or app_type == "code":
                # For Chrome/VS Code, add CDP metadata to relevant elements
                if element.element_type in ["button", "link", "input", "text"]:
                    element.properties["cdp_enhanced"] = True
                    element.properties["app_type"] = app_type

            elif app_type.startswith("libreoffice"):
                # For LibreOffice, add UNO metadata to relevant elements
                if element.element_type in ["table-cell", "paragraph", "slide"]:
                    element.properties["uno_enhanced"] = True
                    element.properties["app_type"] = app_type

        except Exception:
            # Silently fail - enhancement is optional
            pass

    def _map_app_name_to_type(self, app_name: str) -> str:
        """Map application name to app type"""
        app_name_lower = app_name.lower()

        if "code" in app_name_lower or "vscode" in app_name_lower:
            return "code"
        elif "chrome" in app_name_lower or "chromium" in app_name_lower or "google-chrome" in app_name_lower:
            return "chrome"
        elif "calc" in app_name_lower or "spreadsheet" in app_name_lower:
            return "libreoffice_calc"
        elif "writer" in app_name_lower or "document" in app_name_lower:
            return "libreoffice_writer"
        elif "impress" in app_name_lower or "presentation" in app_name_lower:
            return "libreoffice_impress"
        elif "soffice" in app_name_lower or "libreoffice" in app_name_lower:
            return "libreoffice"
        elif "vlc" in app_name_lower or "media" in app_name_lower:
            return "vlc"
        elif "gnome-shell" in app_name_lower or "gjs" in app_name_lower or "desktop" in app_name_lower:
            return "desktop"
        else:
            return "unknown"


class AutoglmElementTracker:
    """Track UI elements using autoglm_v tools"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.grounding_agent = GroundingAgent()
        from perturbation_engine.pipeline.clean_llm_services import CleanElementIdentificationLLM

        self.llm = CleanElementIdentificationLLM()

    def identify_target_element_candidates(
        self, action_str: str, window_states: List[WindowState]
    ) -> List[UIElement]:
        """
        Identify ALL possible target element candidates using LLM-based approach.

        Returns a list of UIElement candidates ranked by likelihood for multi-rollout testing.
        """
        try:
            # Use LLM to identify ALL possible target elements directly with WindowState objects
            llm_candidates = self._identify_candidates_with_llm(action_str, window_states)
            if not llm_candidates:
                self.logger.warning(f"✗ No target element candidates found for: {action_str[:100]}")
                return []

            # Convert LLM candidates to actual UIElement objects
            element_candidates = []
            for llm_candidate in llm_candidates:
                target_element = self._find_element_by_identifier(llm_candidate, window_states)
                if target_element:
                    element_candidates.append(target_element)

            self.logger.info(
                f"✓ Found {len(element_candidates)} target element candidates for: {action_str[:50]}..."
            )
            return element_candidates

        except Exception as e:
            self.logger.error(f"Error identifying target element candidates: {e}")
            return []

    def identify_target_element(
        self, action_str: str, window_states: List[WindowState]
    ) -> Optional[UIElement]:
        """
        Identify single target element (backward compatibility).

        Returns the first (highest confidence) candidate from identify_target_element_candidates.
        """
        candidates = self.identify_target_element_candidates(action_str, window_states)
        return candidates[0] if candidates else None

    def _identify_candidates_with_llm(
        self, action_str: str, window_states: List[WindowState]
    ) -> List[Dict[str, Any]]:
        """Use LLM to identify ALL possible target element candidates"""
        try:
            retries = 0
            while retries < 3:
                retries += 1
                result = self.llm.identify_target_element_candidates(action_str, window_states)
                if result:
                    return result
                else:
                    self.logger.warning(
                        f"LLM failed to identify target element candidates. Retrying ({retries}/3)..."
                    )
            return []

        except Exception as e:
            self.logger.exception(f"Error with LLM element identification: {e}")
            return []

    def _identify_with_llm(
        self, action_str: str, app_states: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Use LLM to identify single target element (backward compatibility)"""
        candidates = self._identify_candidates_with_llm(action_str, app_states)
        return candidates[0] if candidates else None

    def _find_element_by_identifier(
        self, llm_result: Dict[str, Any], window_states: List[WindowState]
    ) -> Optional[UIElement]:
        """Find the actual element in window states using LLM's identifier"""
        try:
            # Extract identifiers from LLM result
            target_name = llm_result.get("name", "").lower()
            target_type = llm_result.get("element_type", "").lower()
            target_app = llm_result.get("app_name", "").lower()

            if not target_name:
                return None

            # Search through all window states
            for window_state in window_states:
                # Check if this is the right app
                if target_app and target_app not in window_state.app_name.lower():
                    continue

                # Search elements in this window
                for element in window_state.get_all_elements(include_structural=False):
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
        self, target_element: UIElement, window_states: List[WindowState]
    ) -> Optional[UIElement]:
        """Track element after perturbation to see if it moved"""
        try:
            # Find element with same properties in new window states
            for window_state in window_states:
                for element in window_state.get_all_elements(include_structural=False):
                    if (
                        element.element_type == target_element.element_type
                        and element.name == target_element.name
                        and element.position == target_element.position
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

    def _map_app_name_to_type(self, app_name: str) -> str:
        """Map application name to app type"""
        app_name_lower = app_name.lower()

        if "code" in app_name_lower or "vscode" in app_name_lower:
            return "code"
        elif "chrome" in app_name_lower or "chromium" in app_name_lower or "google-chrome" in app_name_lower:
            return "chrome"
        elif "calc" in app_name_lower or "spreadsheet" in app_name_lower:
            return "libreoffice_calc"
        elif "writer" in app_name_lower or "document" in app_name_lower:
            return "libreoffice_writer"
        elif "impress" in app_name_lower or "presentation" in app_name_lower:
            return "libreoffice_impress"
        elif "soffice" in app_name_lower or "libreoffice" in app_name_lower:
            return "libreoffice"
        elif "vlc" in app_name_lower or "media" in app_name_lower:
            return "vlc"
        elif "gnome-shell" in app_name_lower or "gjs" in app_name_lower or "desktop" in app_name_lower:
            return "desktop"
        else:
            return "unknown"

    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        import datetime

        return datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
