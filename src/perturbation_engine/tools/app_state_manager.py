import logging
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

from perturbation_engine.pipeline.data_models import UIElement, VisibilityState, WindowState


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


class AppStateExtractor:
    """Enhanced app state extractor with X11, CDP, and UNO integration"""

    def __init__(self, controller=None):
        self.logger = logging.getLogger(__name__)
        self.controller = controller
        self.parser = HierarchicalParser(controller=controller)
        if not controller:
            raise ValueError("AppStateExtractor requires a controller")

    def extract_window_states(self, accessibility_tree: str) -> List[WindowState]:
        """Extract window states from accessibility tree"""
        try:
            # Get X11 window information
            z_order_list = self.controller.get_window_z_order()
            focused_window = self.controller.get_focused_window()
            current_desktop = self.controller.get_current_desktop()

            self.logger.info(f"X11: {len(z_order_list)} windows, focused: {focused_window}")

            # Parse AT-SPI2 tree
            root = ET.fromstring(accessibility_tree)
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

        # Log all applications found
        all_apps = [app_node.get("name", "Unknown") for app_node in root if app_node.tag == "application"]
        self.logger.info(f"AT-SPI2 applications: {all_apps}")

        for app_node in root:
            if app_node.tag != "application":
                continue

            app_name = app_node.get("name", "Unknown")

            # Skip system apps
            if self._should_skip_app(app_name):
                self.logger.info(f"Skipping system app: {app_name}")
                continue

            # Find windows in this application
            app_windows = self._extract_windows_from_app(app_node, app_name)
            windows.extend(app_windows)

        self.logger.info(f"Extracted {len(windows)} windows from {len(all_apps)} applications")
        return windows

    def _extract_windows_from_app(self, app_node: ET.Element, app_name: str) -> List[Dict[str, Any]]:
        """Extract windows from a single application"""
        windows = []

        # Standard AT-SPI2 window types that all applications should use
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

        for child_node in app_node:
            # Only process known window types
            if child_node.tag not in window_types:
                continue

            window_name = child_node.get("name", f"{app_name} Window")

            # Add context for Chrome windows to help with webpage vs browser chrome distinction
            if app_name.lower() in ["chrome", "chromium", "google-chrome"]:
                if "chrome://" in window_name.lower() or "chrome-" in window_name.lower():
                    self.logger.info(f"Found {child_node.tag} window (BROWSER CHROME): {window_name}")
                else:
                    self.logger.info(f"Found {child_node.tag} window (WEBPAGE CONTENT): {window_name}")
            else:
                self.logger.info(f"Found {child_node.tag} window: {window_name}")

            # Parse elements with LibreOffice-specific enhancements
            elements = self.parser._parse_element_tree(
                child_node,
                parent_id=None,
                depth=0,
                parent_visibility=VisibilityState.VISIBLE,
                app_name=app_name,
            )

            # Get window properties
            state_ns = "https://accessibility.ubuntu.example.org/ns/state"
            is_modal = child_node.get(f"{{{state_ns}}}modal") == "true"
            is_active = child_node.get(f"{{{state_ns}}}active") == "true"
            geometry = self.parser._parse_position(child_node) or {}

            # Enhance elements with better interactive element detection
            elements = self._enhance_interactive_elements(elements, child_node, app_name)

            windows.append(
                {
                    "app_name": app_name,
                    "window_name": window_name,
                    "elements": self._flatten_elements(elements) if elements else [],
                    "is_modal": is_modal,
                    "is_active": is_active,
                    "geometry": geometry,
                    "atspi_node": child_node,
                }
            )

        if not windows:
            self.logger.warning(f"No standard windows found in {app_name}")

        return windows

    def _enhance_interactive_elements(
        self, elements: Optional[UIElement], window_node: ET.Element, app_name: str
    ) -> Optional[UIElement]:
        """Enhance elements with better interactive element detection for all applications"""
        if not elements:
            return elements

        # Look for interactive elements that might be missed
        interactive_elements = self._find_interactive_elements(window_node, app_name)
        if interactive_elements:
            # Add interactive elements to the root element
            for interactive_elem in interactive_elements:
                elements.children.append(interactive_elem)

        return elements

    def _find_interactive_elements(self, window_node: ET.Element, app_name: str) -> List[UIElement]:
        """Find interactive elements in application windows"""
        interactive_elements = []

        # Look for common interactive element containers
        interactive_containers = ["menu-bar", "tool-bar", "status-bar", "navigation"]

        for child in window_node:
            if child.tag in interactive_containers:
                # Parse interactive container and its children
                interactive_element = self.parser._parse_element_tree(
                    child,
                    parent_id=None,
                    depth=0,
                    parent_visibility=VisibilityState.VISIBLE,
                    app_name=app_name,
                )
                if interactive_element:
                    interactive_elements.append(interactive_element)

        return interactive_elements

    def _match_x11_with_atspi(
        self,
        atspi_windows: List[Dict[str, Any]],
        z_order_list: List[str],
        focused_window: Optional[str],
        current_desktop: int,
    ) -> List[WindowState]:
        """Match AT-SPI2 windows with X11 windows"""
        window_states = []

        for atspi_window in atspi_windows:
            # Find matching X11 window
            x11_window_id = self._find_matching_x11_window(
                atspi_window["app_name"], atspi_window["window_name"]
            )

            if not x11_window_id:
                # Try LibreOffice-specific matching for --norestore launches
                x11_window_id = self._find_libreoffice_norestore_window(
                    atspi_window["app_name"], atspi_window["window_name"]
                )

            if not x11_window_id:
                # Try Chrome fallback for any unmatched Chrome windows
                x11_window_id = self._find_chrome_window_fallback(
                    atspi_window["app_name"], atspi_window["window_name"]
                )

            if not x11_window_id:
                self.logger.warning(
                    f"No X11 window found for '{atspi_window['window_name']}' from '{atspi_window['app_name']}'"
                )
                continue

            # Get real geometry from X11
            geometry = self.controller.get_window_geometry(x11_window_id)
            if not geometry:
                continue

            # Validate X11 geometry for LibreOffice applications
            if any(
                libreoffice in atspi_window["app_name"].lower()
                for libreoffice in ["libreoffice", "soffice", "calc", "writer", "impress"]
            ):
                # Check if X11 geometry is invalid (too small or unmapped)
                if (
                    geometry.get("width", 0) < 100
                    or geometry.get("height", 0) < 100
                    or not geometry.get("mapped", False)
                ):
                    self.logger.warning(
                        f"Invalid X11 geometry for LibreOffice window {x11_window_id}: {geometry}"
                    )
                    # Use AT-SPI2 geometry as fallback for LibreOffice
                    atspi_geometry = atspi_window.get("geometry", {})
                    if atspi_geometry and atspi_geometry.get("width", 0) > 100:
                        self.logger.info(f"Using AT-SPI2 geometry as fallback: {atspi_geometry}")
                        geometry = atspi_geometry
                    else:
                        self.logger.warning("Skipping LibreOffice window due to invalid geometry")
                        continue

            # Get z-order position
            try:
                z_order = len(z_order_list) - z_order_list.index(x11_window_id)
            except ValueError:
                z_order = 0

            # Get desktop
            window_desktop = self.controller.get_window_desktop(x11_window_id)

            # Create WindowState
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

            # Include windows on current desktop
            if window_desktop == -1 or window_desktop == current_desktop:
                window_states.append(window_state)

        # Sort by z-order (highest first = topmost)
        window_states.sort(key=lambda w: w.z_order, reverse=True)

        self.logger.info(f"Matched {len(window_states)} windows")
        return window_states

    def _find_matching_x11_window(self, app_name: str, window_name: str) -> Optional[str]:
        """Find X11 window ID that matches AT-SPI2 window"""
        try:
            # Try multiple application name variants
            app_variants = self._get_app_name_variants(app_name)

            for variant in app_variants:
                window_ids = self.controller.find_windows_for_app(variant)

                if not window_ids:
                    continue

                # Match by window title
                for window_id in window_ids:
                    x11_title = self.controller.get_window_name(window_id)

                    # Exact match
                    if window_name == x11_title:
                        self.logger.debug(f"Exact match: '{window_name}' = '{x11_title}'")
                        return window_id

                    # Partial match
                    if window_name in x11_title or x11_title in window_name:
                        self.logger.debug(f"Partial match: '{window_name}' in '{x11_title}'")
                        return window_id

                # If only one window for this variant, assume it's the one
                if len(window_ids) == 1:
                    self.logger.debug(f"Single window assumption for {variant}: {window_ids[0]}")
                    return window_ids[0]

            return None

        except Exception as e:
            self.logger.debug(f"Error finding X11 window for {app_name}: {e}")
            return None

    def _get_app_name_variants(self, app_name: str) -> List[str]:
        """Get possible X11 application name variants"""
        variants = [app_name]

        # LibreOffice variants
        if "libreoffice" in app_name.lower() or "calc" in app_name.lower():
            variants.extend(["soffice", "libreoffice-calc", "libreoffice"])
        elif "writer" in app_name.lower():
            variants.extend(["soffice", "libreoffice-writer", "libreoffice"])
        elif "impress" in app_name.lower():
            variants.extend(["soffice", "libreoffice-impress", "libreoffice"])

        # Chrome variants
        elif "chrome" in app_name.lower():
            variants.extend(["google-chrome", "chromium", "chrome"])

        # VS Code variants
        elif "code" in app_name.lower():
            variants.extend(["code", "visual-studio-code"])

        # VLC variants
        elif "vlc" in app_name.lower():
            variants.extend(["vlc", "vlc media player"])

        # Remove duplicates while preserving order
        seen = set()
        unique_variants = []
        for variant in variants:
            if variant not in seen:
                seen.add(variant)
                unique_variants.append(variant)

        return unique_variants

    def _find_libreoffice_norestore_window(self, app_name: str, window_name: str) -> Optional[str]:
        """Find X11 window for LibreOffice --norestore launches where window titles don't match"""
        # Only apply to LibreOffice applications
        if not any(
            libreoffice in app_name.lower()
            for libreoffice in ["libreoffice", "soffice", "calc", "writer", "impress"]
        ):
            return None

        try:
            # Get LibreOffice windows
            variants = self._get_app_name_variants(app_name)
            for variant in variants:
                window_ids = self.controller.find_windows_for_app(variant)

                if not window_ids:
                    continue

                # Look for windows with generic LibreOffice titles
                for window_id in window_ids:
                    x11_title = self.controller.get_window_name(window_id)

                    # Check if this window is mapped (visible) and has geometry
                    try:
                        geometry = self.controller.get_window_geometry(window_id)
                        if geometry and geometry.get("mapped", True):
                            # Match generic LibreOffice titles (version-based or VCL-based)
                            if (
                                "LibreOffice" in x11_title and "VCL" not in x11_title
                            ) or "VCL ImplGetDefaultWindow" in x11_title:
                                self.logger.info(
                                    f"LibreOffice --norestore match: '{window_name}' → '{x11_title}' ({window_id})"
                                )
                                return window_id
                    except Exception as e:
                        self.logger.debug(f"Error checking geometry for window {window_id}: {e}")
                        continue

                # If multiple windows, prefer the main LibreOffice window over VCL windows
                if len(window_ids) > 1:
                    for window_id in window_ids:
                        x11_title = self.controller.get_window_name(window_id)
                        try:
                            geometry = self.controller.get_window_geometry(window_id)
                            if (
                                geometry
                                and geometry.get("mapped", True)
                                and "LibreOffice" in x11_title
                                and "VCL" not in x11_title
                            ):
                                self.logger.info(
                                    f"LibreOffice --norestore fallback: using main window '{x11_title}' ({window_id})"
                                )
                                return window_id
                        except Exception as e:
                            self.logger.debug(f"Error checking geometry for window {window_id}: {e}")
                            continue

                # Last resort: use the first mapped window
                for window_id in window_ids:
                    try:
                        geometry = self.controller.get_window_geometry(window_id)
                        if geometry and geometry.get("mapped", True):
                            x11_title = self.controller.get_window_name(window_id)
                            self.logger.info(
                                f"LibreOffice --norestore fallback: using first mapped window '{x11_title}' ({window_id})"
                            )
                            return window_id
                    except Exception as e:
                        self.logger.debug(f"Error checking geometry for window {window_id}: {e}")
                        continue

            return None

        except Exception as e:
            self.logger.debug(f"Error in LibreOffice --norestore matching: {e}")
            return None

    def _find_chrome_window_fallback(self, app_name: str, window_name: str) -> Optional[str]:
        """Find any Chrome window when exact matching fails - handles popups, settings, dev tools, etc."""
        # Only apply to Chrome applications
        if not any(chrome in app_name.lower() for chrome in ["chrome", "chromium", "google-chrome"]):
            return None

        try:
            # Get Chrome windows
            variants = self._get_app_name_variants(app_name)
            for variant in variants:
                window_ids = self.controller.find_windows_for_app(variant)

                if not window_ids:
                    continue

                # Strategy 1: Look for any valid Chrome window (regardless of size)
                for window_id in window_ids:
                    try:
                        geometry = self.controller.get_window_geometry(window_id)
                        x11_title = self.controller.get_window_name(window_id)

                        # Only check if geometry exists and window is mapped (not hidden)
                        if geometry and geometry.get("mapped", True):
                            self.logger.info(
                                f"Chrome window fallback match: '{window_name}' → '{x11_title}' ({window_id})"
                            )
                            return window_id

                    except Exception as e:
                        self.logger.debug(f"Error checking Chrome window {window_id}: {e}")
                        continue

                # Strategy 2: If no mapped windows found, use any available window
                if window_ids:
                    try:
                        first_window = window_ids[0]
                        x11_title = self.controller.get_window_name(first_window)
                        self.logger.info(
                            f"Chrome window fallback (any available): '{window_name}' → '{x11_title}' ({first_window})"
                        )
                        return first_window
                    except Exception as e:
                        self.logger.debug(f"Error getting first Chrome window: {e}")

            return None

        except Exception as e:
            self.logger.debug(f"Error in Chrome window fallback: {e}")
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
        """Skip system/background apps"""
        # Skip known system/background processes
        skip_patterns = [
            "vmware-user",
            "gsd-",
            "ibus-",
            "evolution-alarm",
            "xdg-desktop-portal",
            "org.gnome.Software",
        ]
        return any(pattern in app_name for pattern in skip_patterns)


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

        # Special handling for interactive elements that might not have positions
        if self._is_interactive_element(node):
            # Interactive elements might not have positions but are still important
            if not position:
                position = self._estimate_interactive_element_position(node, parent_id)

        if not position and not self._is_structural_element(node) and not self._is_interactive_element(node):
            # Skip elements without position unless they're containers or interactive elements
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

    def _is_interactive_element(self, node: ET.Element) -> bool:
        """Check if this is an interactive element that should be preserved"""
        # Interactive element types across all applications
        interactive_types = {
            "menu-bar",
            "menu",
            "menu-item",
            "popup-menu",
            "combo-box",
            "push-button",
            "button",
            "link",
            "input",
            "text",
            "check-box",
            "radio-button",
            "slider",
            "spin-button",
            "tab",
            "tab-item",
            "tree-item",
            "list-item",
            "table-cell",
        }

        return node.tag in interactive_types

    def _estimate_interactive_element_position(
        self, node: ET.Element, parent_id: Optional[str]
    ) -> Optional[Dict[str, int]]:
        """Estimate position for interactive elements that don't have explicit coordinates"""
        # For interactive elements, we can't reliably estimate position without more context
        # Return None to let the system handle it
        return None

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
        """Extract position and size using consistent regex parsing"""
        try:
            coords_str = node.get(f"{{{self.component_ns}}}screencoord", "")
            size_str = node.get(f"{{{self.component_ns}}}size", "")

            if not coords_str or not size_str:
                return None

            # Use regex parsing for consistency with extract_coordinate_from_node
            coords_match = re.match(r"\((\d+), (\d+)\)", coords_str)
            size_match = re.match(r"\((\d+), (\d+)\)", size_str)

            if not coords_match or not size_match:
                return None

            x, y = int(coords_match.group(1)), int(coords_match.group(2))
            w, h = int(size_match.group(1)), int(size_match.group(2))

            if x < 0 or y < 0 or w <= 0 or h <= 0:
                return None

            return {
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "center_x": x + w // 2,
                "center_y": y + h // 2,
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


class ElementValidator:
    """Validate UI element accessibility and reachability"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def validate_element_reachability(
        self, target_element: UIElement, window_states: List[WindowState], perturbation_command: str = None
    ) -> Tuple[bool, str]:
        """Validate that target elements remain reachable after perturbation"""
        if not target_element:
            return False, "No target element provided"

        # Run comprehensive element validation checks
        validation_checks = [
            self._check_element_visibility(target_element),
            self._check_element_accessibility(target_element),
            self._check_element_z_order_blocking(target_element, window_states),
            self._check_perturbation_blocking(perturbation_command, target_element)
            if perturbation_command
            else (True, ""),
        ]

        for is_valid, reason in validation_checks:
            if not is_valid:
                return False, reason

        return True, "Target element is reachable"

    def _check_element_visibility(self, element: UIElement) -> Tuple[bool, str]:
        """Check if element is visible"""
        # Check visibility states that make elements inaccessible
        inaccessible_states = [
            VisibilityState.HIDDEN_COLLAPSED,
            VisibilityState.HIDDEN_WINDOW,
            VisibilityState.HIDDEN_TAB,
            VisibilityState.HIDDEN_NOT_SHOWING,
        ]

        if element.visibility in inaccessible_states:
            return False, f"Element is not visible: {element.visibility.value}"

        return True, "Element is visible"

    def _check_element_accessibility(self, element: UIElement) -> Tuple[bool, str]:
        """Check if element is accessible"""
        # Check if element is enabled
        if not element.is_enabled:
            return False, "Element is disabled"

        return True, "Element is accessible"

    def _check_element_z_order_blocking(
        self, element: UIElement, window_states: List[WindowState]
    ) -> Tuple[bool, str]:
        """Check if element is blocked by z-order"""
        if self._is_element_blocked_by_z_order(element, window_states):
            return False, "Element is blocked by higher z-order window"

        return True, "Element is not blocked by z-order"

    def _check_perturbation_blocking(
        self, perturbation_command: str, target_element: UIElement
    ) -> Tuple[bool, str]:
        """Check if perturbation would block element"""
        if self._would_perturbation_block_element(perturbation_command, target_element):
            return False, "Perturbation would make target element unreachable"

        return True, "Perturbation will not block element"

    def _is_element_blocked_by_z_order(self, element: UIElement, window_states: List[WindowState]) -> bool:
        """Check if element is blocked by higher z-order windows"""
        if not element.position:
            return False

        element_x = element.position.get("x", 0)
        element_y = element.position.get("y", 0)
        element_width = element.position.get("width", 0)
        element_height = element.position.get("height", 0)

        if element_width <= 0 or element_height <= 0:
            return False

        # Find the window containing this element
        element_window = self._find_element_window(element, window_states)
        if not element_window:
            return False

        # Check if any higher z-order window overlaps with element
        element_z_order = element_window.z_order

        for window_state in window_states:
            if self._is_higher_z_order_window(window_state, element_z_order):
                if self._windows_overlap(window_state, element_x, element_y, element_width, element_height):
                    return True

        return False

    def _find_element_window(
        self, element: UIElement, window_states: List[WindowState]
    ) -> Optional[WindowState]:
        """Find the window containing the given element"""
        for window_state in window_states:
            if window_state.root_element and self._element_in_window(element, window_state.root_element):
                return window_state
        return None

    def _is_higher_z_order_window(self, window_state: WindowState, element_z_order: int) -> bool:
        """Check if window has higher z-order than element"""
        return window_state.z_order > element_z_order and window_state.is_mapped

    def _windows_overlap(
        self,
        window_state: WindowState,
        element_x: int,
        element_y: int,
        element_width: int,
        element_height: int,
    ) -> bool:
        """Check if window overlaps with element coordinates"""
        if not window_state.geometry:
            return False

        window_x = window_state.geometry.get("x", 0)
        window_y = window_state.geometry.get("y", 0)
        window_width = window_state.geometry.get("width", 0)
        window_height = window_state.geometry.get("height", 0)

        return (
            window_x < element_x + element_width
            and window_x + window_width > element_x
            and window_y < element_y + element_height
            and window_y + window_height > element_y
        )

    def _element_in_window(self, element: UIElement, window_root: UIElement) -> bool:
        """Check if element is within a window's element tree"""
        if not element or not window_root:
            return False

        # Simple check - in a real implementation, you'd traverse the tree
        return True  # Placeholder - assume element is in window

    def _would_perturbation_block_element(self, perturbation_command: str, target_element: UIElement) -> bool:
        """Check if perturbation command would block the target element"""
        command_lower = perturbation_command.lower()

        # Commands that could block elements
        blocking_commands = [
            "hide",
            "display: none",
            "visibility: hidden",
            "opacity: 0",
            "z-index: -1",
            "position: absolute",
            "left: -9999px",
            "width: 0",
            "height: 0",
            "minimize",
            "close",
        ]

        for blocking_cmd in blocking_commands:
            if blocking_cmd in command_lower:
                return True

        return False


class ElementTracker:
    """Track UI elements"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        from perturbation_engine.pipeline.llm_services import ElementIdentificationLLM

        self.llm = ElementIdentificationLLM()
        self.validator = ElementValidator()

    def identify_target_element_candidates(
        self, action_str: str, window_states: List[WindowState]
    ) -> List[UIElement]:
        """
        Identify ALL possible target element candidates using LLM-based approach.

        Returns a list of UIElement candidates ranked by likelihood for multi-rollout testing.
        """
        try:
            # Store window states for screen size detection
            self._last_window_states = window_states

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
            target_name = llm_result.get("name", "").strip()
            target_type = llm_result.get("element_type", "").strip()
            target_app = llm_result.get("app_name", "").strip()

            if not target_name:
                return None

            # Find the right window first
            target_window = None
            for window_state in window_states:
                if target_app and target_app.lower() in window_state.app_name.lower():
                    target_window = window_state
                    break

            if not target_window:
                # If no app match, use the first active window
                target_window = next(
                    (w for w in window_states if w.is_active), window_states[0] if window_states else None
                )

            if not target_window:
                return None

            # Simple search: look for elements that contain the target name
            target_lower = target_name.lower()

            for element in target_window.get_all_elements(include_structural=False):
                if not element.name:
                    continue

                element_name_lower = element.name.lower()

                # Direct substring match
                if target_lower in element_name_lower:
                    # Check type if specified
                    if target_type and target_type.lower() != element.element_type.lower():
                        continue

                    # Add metadata and return
                    element.properties.update(
                        {
                            "llm_identified": True,
                            "llm_confidence": llm_result.get("confidence", 1.0),
                            "llm_reasoning": llm_result.get("reasoning", ""),
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
        """Update action coordinates to the exact center of the new position.

        Handles both OSWorld-Human format (natural language) and pyautogui format (with coordinates).
        For OSWorld-Human format, converts to pyautogui format with new coordinates.
        For pyautogui format, updates existing coordinates.

        Supports all action types from execute_action(): MOVE_TO, CLICK, MOUSE_DOWN, MOUSE_UP,
        RIGHT_CLICK, DOUBLE_CLICK, DRAG_TO, SCROLL, TYPING, PRESS, KEY_DOWN, KEY_UP, HOTKEY
        """

        new_x = new_position["center_x"]
        new_y = new_position["center_y"]

        # Check if this is an OSWorld-Human format action (natural language)
        if action_str.startswith("`") and any(
            action_type in action_str
            for action_type in [
                "CLICK",
                "TYPING",
                "PRESS",
                "MOVE_TO",
                "MOUSE_DOWN",
                "MOUSE_UP",
                "RIGHT_CLICK",
                "DOUBLE_CLICK",
                "TRIPLE_CLICK",
                "DRAG_TO",
                "SCROLL",
                "KEY_DOWN",
                "KEY_UP",
                "HOTKEY",
            ]
        ):
            # Convert OSWorld-Human format to pyautogui format with coordinates
            if "`CLICK`" in action_str:
                target_desc = action_str.replace("`CLICK`", "").strip()
                updated_action = f"pyautogui.click({new_x}, {new_y})  # {target_desc}"
            elif "`RIGHT_CLICK`" in action_str:
                target_desc = action_str.replace("`RIGHT_CLICK`", "").strip()
                updated_action = f"pyautogui.rightClick({new_x}, {new_y})  # {target_desc}"
            elif "`DOUBLE_CLICK`" in action_str:
                target_desc = action_str.replace("`DOUBLE_CLICK`", "").strip()
                updated_action = f"pyautogui.doubleClick({new_x}, {new_y})  # {target_desc}"
            elif "`TRIPLE_CLICK`" in action_str:
                target_desc = action_str.replace("`TRIPLE_CLICK`", "").strip()
                updated_action = f"pyautogui.click({new_x}, {new_y}, clicks=3)  # {target_desc}"
            elif "`MOVE_TO`" in action_str:
                target_desc = action_str.replace("`MOVE_TO`", "").strip()
                updated_action = f"pyautogui.moveTo({new_x}, {new_y})  # {target_desc}"
            elif "`DRAG_TO`" in action_str:
                target_desc = action_str.replace("`DRAG_TO`", "").strip()
                updated_action = f"pyautogui.dragTo({new_x}, {new_y}, duration=1.0, button='left', mouseDownUp=True)  # {target_desc}"
            elif "`TYPING`" in action_str:
                # Extract the text to type - handle various formats
                # Format 1: `TYPING` 'text'
                # Format 2: `TYPING` "text"
                # Format 3: `TYPING` text (no quotes)
                typing_match = re.search(r"`TYPING`\s*(?:['\"]([^'\"]+)['\"]|(.+))", action_str)
                if typing_match:
                    # Try quoted text first, then unquoted text
                    text_to_type = typing_match.group(1) or typing_match.group(2)
                    if text_to_type:
                        text_to_type = text_to_type.strip()
                        updated_action = f"pyautogui.typewrite({repr(text_to_type)})  # {action_str}"
                    else:
                        updated_action = f"pyautogui.typewrite('')  # {action_str}"
                else:
                    updated_action = f"pyautogui.typewrite('')  # {action_str}"
            elif "`PRESS`" in action_str:
                # Extract the key to press - handle various formats
                # Format 1: `PRESS` 'key'
                # Format 2: `PRESS` "key"
                # Format 3: `PRESS` key (no quotes)
                press_match = re.search(r"`PRESS`\s*(?:['\"]([^'\"]+)['\"]|(.+))", action_str)
                if press_match:
                    # Try quoted key first, then unquoted key
                    key_to_press = press_match.group(1) or press_match.group(2)
                    if key_to_press:
                        key_to_press = key_to_press.strip()
                        updated_action = f"pyautogui.press({repr(key_to_press)})  # {action_str}"
                    else:
                        updated_action = f"pyautogui.press('')  # {action_str}"
                else:
                    updated_action = f"pyautogui.press('')  # {action_str}"
            elif "`KEY_DOWN`" in action_str:
                # Extract the key - handle various formats
                key_match = re.search(r"`KEY_DOWN`\s*(?:['\"]([^'\"]+)['\"]|(.+))", action_str)
                if key_match:
                    key = key_match.group(1) or key_match.group(2)
                    if key:
                        key = key.strip()
                        updated_action = f"pyautogui.keyDown({repr(key)})  # {action_str}"
                    else:
                        updated_action = f"pyautogui.keyDown('')  # {action_str}"
                else:
                    updated_action = f"pyautogui.keyDown('')  # {action_str}"
            elif "`KEY_UP`" in action_str:
                # Extract the key - handle various formats
                key_match = re.search(r"`KEY_UP`\s*(?:['\"]([^'\"]+)['\"]|(.+))", action_str)
                if key_match:
                    key = key_match.group(1) or key_match.group(2)
                    if key:
                        key = key.strip()
                        updated_action = f"pyautogui.keyUp({repr(key)})  # {action_str}"
                    else:
                        updated_action = f"pyautogui.keyUp('')  # {action_str}"
                else:
                    updated_action = f"pyautogui.keyUp('')  # {action_str}"
            elif "`HOTKEY`" in action_str:
                # Extract keys from HOTKEY action - handle multiple formats
                # Format 1: `HOTKEY` 'Ctrl-P' (single key combination)
                # Format 2: `HOTKEY` ['ctrl', 'c'] (multiple keys in brackets)
                # Format 3: `HOTKEY` 'ctrl', 'c' (multiple keys separated by comma)

                # Try format 1: single key combination like 'Ctrl-P'
                single_key_match = re.search(r"`HOTKEY`\s*['\"]([^'\"]+)['\"]", action_str)
                if single_key_match:
                    key_combo = single_key_match.group(1)
                    # Split key combination like 'Ctrl-P' into ['ctrl', 'p']
                    if "-" in key_combo:
                        keys = [key.strip().lower() for key in key_combo.split("-")]
                        keys_repr = ", ".join([repr(key) for key in keys])
                        updated_action = f"pyautogui.hotkey({keys_repr})  # {action_str}"
                    else:
                        updated_action = f"pyautogui.hotkey({repr(key_combo.lower())})  # {action_str}"
                else:
                    # Try format 2: multiple keys in brackets
                    bracket_match = re.search(r"`HOTKEY`\s*\[([^\]]+)\]", action_str)
                    if bracket_match:
                        keys_str = bracket_match.group(1)
                        # Parse keys from string like "ctrl, c" or "'ctrl', 'c'"
                        keys = [key.strip().strip("'\"") for key in keys_str.split(",")]
                        keys_repr = ", ".join([repr(key) for key in keys])
                        updated_action = f"pyautogui.hotkey({keys_repr})  # {action_str}"
                    else:
                        # Try format 3: multiple keys separated by comma
                        comma_match = re.search(
                            r"`HOTKEY`\s*['\"]([^'\"]+)['\"],\s*['\"]([^'\"]+)['\"]", action_str
                        )
                        if comma_match:
                            key1, key2 = comma_match.groups()
                            updated_action = f"pyautogui.hotkey({repr(key1)}, {repr(key2)})  # {action_str}"
                        else:
                            updated_action = f"pyautogui.hotkey('')  # {action_str}"
            elif "`SCROLL`" in action_str:
                # Extract scroll parameters
                scroll_match = re.search(r"`SCROLL`\s*dx[:\s]*(-?\d+)[,\s]*dy[:\s]*(-?\d+)", action_str)
                if scroll_match:
                    dx, dy = scroll_match.groups()
                    updated_action = f"pyautogui.hscroll({dx}); pyautogui.vscroll({dy})  # {action_str}"
                else:
                    updated_action = f"pyautogui.scroll(0)  # {action_str}"
            elif "`MOUSE_DOWN`" in action_str:
                # Extract the button - handle various formats
                button_match = re.search(r"`MOUSE_DOWN`\s*(?:['\"]([^'\"]+)['\"]|(.+))", action_str)
                if button_match:
                    button = button_match.group(1) or button_match.group(2)
                    if button:
                        button = button.strip()
                        updated_action = f"pyautogui.mouseDown(button={repr(button)})  # {action_str}"
                    else:
                        updated_action = f"pyautogui.mouseDown()  # {action_str}"
                else:
                    updated_action = f"pyautogui.mouseDown()  # {action_str}"
            elif "`MOUSE_UP`" in action_str:
                # Extract the button - handle various formats
                button_match = re.search(r"`MOUSE_UP`\s*(?:['\"]([^'\"]+)['\"]|(.+))", action_str)
                if button_match:
                    button = button_match.group(1) or button_match.group(2)
                    if button:
                        button = button.strip()
                        updated_action = f"pyautogui.mouseUp(button={repr(button)})  # {action_str}"
                    else:
                        updated_action = f"pyautogui.mouseUp()  # {action_str}"
                else:
                    updated_action = f"pyautogui.mouseUp()  # {action_str}"
            else:
                # Fallback for other action types
                updated_action = f"# {action_str} - converted to coordinates ({new_x}, {new_y})"
        else:
            # Handle pyautogui format - update existing coordinates
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
