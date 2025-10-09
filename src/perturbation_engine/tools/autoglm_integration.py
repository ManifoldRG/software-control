"""
Autoglm_v Integration: Clean interface for autoglm_v tools
Provides app state extraction, element identification, and coordinate tracking
"""

import logging
import re
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from perturbation_engine.pipeline.data_models import AppElement, AppState

# Import existing autoglm_v logic
from perturbation_engine.tools.autoglm_v.prompt.grounding_agent import GroundingAgent
from perturbation_engine.tools.autoglm_v.tools.package.code import CodeTools
from perturbation_engine.tools.autoglm_v.tools.package.google_chrome import BrowserTools
from perturbation_engine.tools.autoglm_v.tools.package.vlc import VLCTools

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


class VisibilityState(Enum):
    """Element visibility states"""

    VISIBLE = "visible"  # Fully visible and interactable
    HIDDEN_COLLAPSED = "collapsed"  # In collapsed menu/dropdown
    HIDDEN_WINDOW = "hidden_window"  # In hidden/minimized window
    HIDDEN_TAB = "hidden_tab"  # In inactive tab
    STRUCTURAL = "structural"  # Container element (frame, panel)
    HIDDEN_NOT_SHOWING = "not_showing"  # AT-SPI2 showing=false


@dataclass
class UIElement:
    """Represents a UI element with hierarchy"""

    element_id: str
    element_type: str
    name: str
    position: Dict[str, int]

    # Hierarchy
    parent_id: Optional[str] = None
    children: List["UIElement"] = field(default_factory=list)
    depth: int = 0

    # States
    visibility: VisibilityState = VisibilityState.VISIBLE
    is_enabled: bool = True
    is_focused: bool = False
    is_expanded: bool = False  # For menus, dropdowns, etc.

    # Additional properties
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to AppElement-compatible dict"""
        return {
            "element_id": self.element_id,
            "element_type": self.element_type,
            "name": self.name,
            "text": self.name,
            "position": self.position,
            "properties": {
                **self.properties,
                "parent_id": self.parent_id,
                "depth": self.depth,
                "visibility": self.visibility.value,
                "is_enabled": self.is_enabled,
                "is_focused": self.is_focused,
                "is_expanded": self.is_expanded,
                "children_count": len(self.children),
            },
        }


@dataclass
class WindowState:
    """Represents a window with its elements and X11 window manager data"""

    window_id: str
    window_name: str
    app_name: str

    # Window properties
    is_active: bool = False
    is_modal: bool = False
    is_minimized: bool = False
    geometry: Dict[str, int] = field(default_factory=dict)
    z_order: int = 0

    # X11 window manager data
    x11_window_id: Optional[str] = None
    is_mapped: bool = True  # Actually visible from X11
    desktop: int = 0  # Virtual desktop number

    # Elements tree
    root_element: Optional[UIElement] = None

    def get_all_elements(self, include_structural: bool = False) -> List[UIElement]:
        """Get flat list of all elements (DFS traversal)"""
        if not self.root_element:
            return []

        elements = []

        def traverse(elem: UIElement):
            # Filter based on visibility
            if elem.visibility == VisibilityState.VISIBLE:
                if include_structural or elem.element_type not in ["frame", "panel", "filler"]:
                    elements.append(elem)

            # Always traverse children (they might be visible even if parent is structural)
            for child in elem.children:
                traverse(child)

        traverse(self.root_element)
        return elements


# Enhanced app-specific enhancement with CDP/UNO integration
class ChromeDPClient:
    """Enhanced Chrome DevTools Protocol client with WebSocket connections"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ws_connections = {}  # port -> websocket
        self.connected_ports = set()

    def connect(self, port: int = 9222) -> bool:
        """Connect to Chrome/VS Code debugging port via WebSocket"""
        try:
            import requests
            import websocket

            # Check if port is available
            response = requests.get(f"http://localhost:{port}/json", timeout=2)
            targets = response.json()

            if not targets:
                return False

            # Connect to first target
            ws_url = targets[0]["webSocketDebuggerUrl"]
            ws = websocket.create_connection(ws_url, timeout=5)
            self.ws_connections[port] = ws
            self.connected_ports.add(port)

            self.logger.info(f"Connected to CDP on port {port}")
            return True

        except Exception as e:
            self.logger.debug(f"Failed to connect to CDP on port {port}: {e}")
            return False

    def get_dom_snapshot(self, port: int = 9222) -> Optional[List[UIElement]]:
        """Get DOM elements as UIElements with precise positioning"""
        try:
            ws = self.ws_connections.get(port)
            if not ws:
                if not self.connect(port):
                    return None
                ws = self.ws_connections[port]

            import json

            # Enable DOM domain
            ws.send(json.dumps({"id": 1, "method": "DOM.enable"}))
            ws.recv()

            # Get document
            ws.send(json.dumps({"id": 2, "method": "DOM.getDocument", "params": {"depth": -1}}))
            response = json.loads(ws.recv())

            if "result" not in response:
                return None

            # Parse DOM tree
            root_node = response["result"]["root"]
            elements = self._parse_dom_node(root_node, ws, port)

            return elements

        except Exception as e:
            self.logger.debug(f"Failed to get DOM snapshot: {e}")
            return None

    def _parse_dom_node(self, node: Dict, ws, port: int, elements: List[UIElement] = None) -> List[UIElement]:
        """Recursively parse DOM node with precise positioning"""
        if elements is None:
            elements = []

        import json

        node_id = node.get("nodeId")
        node_name = node.get("nodeName", "").lower()

        # Only keep interactive elements
        if node_name in ["button", "input", "select", "textarea", "a", "label", "div", "span"]:
            try:
                # Get box model for precise position
                ws.send(
                    json.dumps(
                        {"id": 100 + node_id, "method": "DOM.getBoxModel", "params": {"nodeId": node_id}}
                    )
                )
                box_response = json.loads(ws.recv())

                if "result" in box_response and "model" in box_response["result"]:
                    model = box_response["result"]["model"]
                    content = model.get("content", [0, 0, 0, 0, 0, 0, 0, 0])

                    # Extract bounding box
                    x = int(min(content[0], content[2], content[4], content[6]))
                    y = int(min(content[1], content[3], content[5], content[7]))
                    width = int(max(content[0], content[2], content[4], content[6]) - x)
                    height = int(max(content[1], content[3], content[5], content[7]) - y)

                    if width > 0 and height > 0:
                        # Get text content and attributes
                        attributes = node.get("attributes", [])
                        attr_dict = {attributes[i]: attributes[i + 1] for i in range(0, len(attributes), 2)}

                        name = (
                            attr_dict.get("aria-label", "")
                            or attr_dict.get("title", "")
                            or attr_dict.get("value", "")
                            or attr_dict.get("placeholder", "")
                            or node.get("nodeValue", "")
                        )

                        # Only include elements with meaningful content or interaction
                        if name.strip() or node_name in ["button", "input", "select", "textarea", "a"]:
                            element = UIElement(
                                element_id=f"cdp_{node_id}",
                                element_type=node_name,
                                name=name.strip() or f"{node_name}_{node_id}",
                                position={
                                    "x": x,
                                    "y": y,
                                    "width": width,
                                    "height": height,
                                    "center_x": x + width // 2,
                                    "center_y": y + height // 2,
                                },
                                visibility=VisibilityState.VISIBLE,
                                is_enabled=True,
                                properties={
                                    "role": attr_dict.get("role", node_name),
                                    "href": attr_dict.get("href", ""),
                                    "type": attr_dict.get("type", ""),
                                    "class": attr_dict.get("class", ""),
                                    "id": attr_dict.get("id", ""),
                                    "cdp_node_id": node_id,
                                },
                            )
                            elements.append(element)
            except Exception as e:
                self.logger.debug(f"Failed to parse DOM node {node_id}: {e}")

        # Recurse into children
        for child in node.get("children", []):
            self._parse_dom_node(child, ws, port, elements)

        return elements

    def close(self):
        """Close all WebSocket connections"""
        for ws in self.ws_connections.values():
            try:
                ws.close()
            except Exception:
                pass
        self.ws_connections.clear()
        self.connected_ports.clear()


class LibreOfficeUNOClient:
    """Enhanced LibreOffice UNO API client with better connection management"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.desktop = None
        self.context = None
        self.connected = False

    def connect(self) -> bool:
        """Connect to LibreOffice instance with better error handling"""
        try:
            import uno

            local_context = uno.getComponentContext()
            resolver = local_context.ServiceManager.createInstanceWithContext(
                "com.sun.star.bridge.UnoUrlResolver", local_context
            )

            # Try to connect (LibreOffice must be started with --accept)
            self.context = resolver.resolve(
                "uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext"
            )
            smgr = self.context.ServiceManager
            self.desktop = smgr.createInstanceWithContext("com.sun.star.frame.Desktop", self.context)

            self.connected = True
            self.logger.info("Connected to LibreOffice UNO")
            return True

        except Exception as e:
            self.logger.debug(f"Failed to connect to LibreOffice UNO: {e}")
            self.connected = False
            return False

    def get_calc_cells(self) -> Optional[List[UIElement]]:
        """Extract visible cells from Calc spreadsheet with better positioning"""
        try:
            if not self.connected:
                if not self.connect():
                    return None

            doc = self.desktop.getCurrentComponent()

            if not doc or not hasattr(doc, "getSheets"):
                return None

            elements = []
            _sheets = doc.getSheets()

            # Get active sheet
            controller = doc.getCurrentController()
            active_sheet = controller.getActiveSheet()

            # Get used range on active sheet
            cursor = active_sheet.createCursor()
            cursor.gotoStartOfUsedArea(False)
            cursor.gotoEndOfUsedArea(True)

            range_addr = cursor.getRangeAddress()

            # Extract cells (limit to visible area + some padding)
            for row in range(range_addr.StartRow, min(range_addr.EndRow + 1, range_addr.StartRow + 50)):
                for col in range(
                    range_addr.StartColumn, min(range_addr.EndColumn + 1, range_addr.StartColumn + 20)
                ):
                    cell = active_sheet.getCellByPosition(col, row)

                    cell_value = self._get_cell_value(cell)
                    if cell_value:
                        # Better position estimation (could be improved with actual viewport)
                        x = 100 + col * 80
                        y = 200 + row * 25

                        element = UIElement(
                            element_id=f"calc_cell_{col}_{row}",
                            element_type="table-cell",
                            name=str(cell_value),
                            position={
                                "x": x,
                                "y": y,
                                "width": 80,
                                "height": 25,
                                "center_x": x + 40,
                                "center_y": y + 12,
                            },
                            visibility=VisibilityState.VISIBLE,
                            properties={
                                "value": str(cell_value),
                                "role": f"R{row + 1}C{col + 1}",
                                "column": col,
                                "row": row,
                                "cell_type": str(type(cell_value).__name__),
                            },
                        )
                        elements.append(element)

            return elements

        except Exception as e:
            self.logger.debug(f"Failed to get Calc cells: {e}")
            return None

    def get_writer_paragraphs(self) -> Optional[List[UIElement]]:
        """Extract paragraphs from Writer document"""
        try:
            if not self.connected:
                if not self.connect():
                    return None

            doc = self.desktop.getCurrentComponent()

            if not doc or not hasattr(doc, "getText"):
                return None

            text = doc.getText()
            elements = []

            enum = text.createEnumeration()
            row = 0

            while enum.hasMoreElements():
                para = enum.nextElement()

                if hasattr(para, "getString"):
                    para_text = para.getString()
                    if para_text.strip():
                        element = UIElement(
                            element_id=f"writer_para_{row}",
                            element_type="paragraph",
                            name=para_text[:100],  # Limit length
                            position={
                                "x": 100,
                                "y": 200 + row * 20,
                                "width": 600,
                                "height": 20,
                                "center_x": 400,
                                "center_y": 210 + row * 20,
                            },
                            visibility=VisibilityState.VISIBLE,
                            properties={"value": para_text, "length": len(para_text), "paragraph_index": row},
                        )
                        elements.append(element)
                        row += 1

                if row > 100:  # Limit to first 100 paragraphs
                    break

            return elements

        except Exception as e:
            self.logger.debug(f"Failed to get Writer paragraphs: {e}")
            return None

    def get_impress_slides(self) -> Optional[List[UIElement]]:
        """Extract slides from Impress presentation"""
        try:
            if not self.connected:
                if not self.connect():
                    return None

            doc = self.desktop.getCurrentComponent()

            if not doc or not hasattr(doc, "getDrawPages"):
                return None

            slides = doc.getDrawPages()
            elements = []

            for i in range(min(slides.getCount(), 20)):  # Limit to 20 slides
                _slide = slides.getByIndex(i)

                element = UIElement(
                    element_id=f"slide_{i}",
                    element_type="slide",
                    name=f"Slide {i + 1}",
                    position={
                        "x": 100 + (i % 5) * 150,
                        "y": 200 + (i // 5) * 100,
                        "width": 140,
                        "height": 90,
                        "center_x": 170 + (i % 5) * 150,
                        "center_y": 245 + (i // 5) * 100,
                    },
                    visibility=VisibilityState.VISIBLE,
                    properties={"role": "slide", "slide_index": i, "slide_number": i + 1},
                )
                elements.append(element)

            return elements

        except Exception as e:
            self.logger.debug(f"Failed to get Impress slides: {e}")
            return None

    def _get_cell_value(self, cell):
        """Get cell value regardless of type with better error handling"""
        try:
            cell_type = cell.getType()
            if cell_type.value == "TEXT":
                return cell.getString()
            elif cell_type.value == "VALUE":
                return cell.getValue()
            elif cell_type.value == "FORMULA":
                return cell.getFormula()
            elif cell_type.value == "EMPTY":
                return None
            return None
        except Exception as e:
            self.logger.debug(f"Failed to get cell value: {e}")
            return None


class AppEnhancer:
    """Enhanced app-specific enhancement with improved CDP and UNO integration"""

    def __init__(self, controller=None):
        self.logger = logging.getLogger(__name__)
        self.controller = controller
        self.cdp_client = ChromeDPClient()
        self.uno_client = LibreOfficeUNOClient()

    def enhance_chrome_elements(self, base_elements: List[UIElement]) -> List[UIElement]:
        """Enhance Chrome elements with improved CDP client"""
        try:
            # Try multiple Chrome debugging ports
            cdp_elements = None
            for port in [9222, 9223, 9224]:
                cdp_elements = self.cdp_client.get_dom_snapshot(port)
                if cdp_elements:
                    self.logger.info(
                        f"Enhanced Chrome with {len(cdp_elements)} CDP elements from port {port}"
                    )
                    break

            if cdp_elements:
                # Merge CDP elements with AT-SPI elements
                enhanced_elements = base_elements + cdp_elements
                self.logger.info(
                    f"Enhanced Chrome elements: {len(base_elements)} AT-SPI + {len(cdp_elements)} CDP = {len(enhanced_elements)} total"
                )
                return enhanced_elements
            else:
                self.logger.debug("No CDP elements found for Chrome")
                return base_elements

        except Exception as e:
            self.logger.error(f"Error enhancing Chrome elements: {e}")
            return base_elements

    def enhance_vscode_elements(self, base_elements: List[UIElement]) -> List[UIElement]:
        """Enhance VS Code elements with improved CDP client"""
        try:
            # Try VS Code debugging ports
            cdp_elements = None
            for port in [9222, 9229]:
                cdp_elements = self.cdp_client.get_dom_snapshot(port)
                if cdp_elements:
                    self.logger.info(
                        f"Enhanced VS Code with {len(cdp_elements)} CDP elements from port {port}"
                    )
                    break

            if cdp_elements:
                # Merge CDP elements with AT-SPI elements
                enhanced_elements = base_elements + cdp_elements
                self.logger.info(
                    f"Enhanced VS Code elements: {len(base_elements)} AT-SPI + {len(cdp_elements)} CDP = {len(enhanced_elements)} total"
                )
                return enhanced_elements
            else:
                self.logger.debug("No CDP elements found for VS Code")
                return base_elements

        except Exception as e:
            self.logger.error(f"Error enhancing VS Code elements: {e}")
            return base_elements

    def enhance_libreoffice_elements(self, base_elements: List[UIElement], app_type: str) -> List[UIElement]:
        """Enhance LibreOffice elements with improved UNO client"""
        uno_elements = None

        if app_type == "libreoffice_calc":
            uno_elements = self.uno_client.get_calc_cells()
        elif app_type == "libreoffice_writer":
            uno_elements = self.uno_client.get_writer_paragraphs()
        elif app_type == "libreoffice_impress":
            uno_elements = self.uno_client.get_impress_slides()

        if uno_elements:
            # Merge UNO elements with AT-SPI elements
            enhanced_elements = base_elements + uno_elements
            self.logger.info(
                f"Enhanced LibreOffice {app_type}: {len(base_elements)} AT-SPI + {len(uno_elements)} UNO = {len(enhanced_elements)} total"
            )
            return enhanced_elements
        else:
            self.logger.debug(f"No UNO elements found for LibreOffice {app_type}")
            return base_elements

    def cleanup(self):
        """Clean up CDP and UNO connections"""
        try:
            if hasattr(self, "cdp_client"):
                self.cdp_client.close()
            if hasattr(self, "uno_client"):
                # UNO connections are typically cleaned up automatically
                pass
            self.logger.info("AppEnhancer cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during AppEnhancer cleanup: {e}")

    def __del__(self):
        """Cleanup connections on destruction"""
        try:
            self.cleanup()
        except Exception:
            pass


class X11WindowManager:
    """Get REAL window state from X11 window manager with comprehensive fallback"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._x11_available = self._check_x11_tools()
        self._available_tools = self._check_available_tools()

    def _check_x11_tools(self) -> bool:
        """Check if X11 tools are available"""
        try:
            subprocess.run(["xprop", "--version"], capture_output=True, timeout=1)
            subprocess.run(["xdotool", "--version"], capture_output=True, timeout=1)
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.logger.warning("X11 tools (xprop, xdotool) not available - using AT-SPI2 fallback")
            return False

    def _check_available_tools(self) -> Dict[str, bool]:
        """Check which specific X11 tools are available"""
        tools = ["xprop", "xdotool", "xwininfo", "wmctrl"]
        available = {}

        for tool in tools:
            try:
                subprocess.run([tool, "--version"], capture_output=True, timeout=1)
                available[tool] = True
            except (subprocess.TimeoutExpired, FileNotFoundError):
                available[tool] = False

        self.logger.info(f"X11 tools availability: {available}")
        return available

    def get_window_z_order(self) -> List[str]:
        """Get actual window stacking order from window manager with fallback"""
        if not self._available_tools.get("xprop", False):
            self.logger.debug("xprop not available for z-order detection")
            return []

        try:
            result = subprocess.run(
                ["xprop", "-root", "_NET_CLIENT_LIST_STACKING"], capture_output=True, text=True, timeout=2
            )

            if result.returncode != 0:
                return []

            # Parse: _NET_CLIENT_LIST_STACKING(WINDOW): window id # 0x3400001, 0x3400002, ...
            window_ids = re.findall(r"0x[0-9a-f]+", result.stdout)

            # List is bottom-to-top, so reverse for top-to-bottom
            return list(reversed(window_ids))

        except Exception as e:
            self.logger.debug(f"Failed to get z-order: {e}")
            return []

    def get_window_geometry(self, window_id: str) -> Optional[Dict[str, int]]:
        """Get window position and size from xwininfo with fallback"""
        if not self._available_tools.get("xwininfo", False):
            self.logger.debug("xwininfo not available for geometry detection")
            return None

        try:
            result = subprocess.run(
                ["xwininfo", "-id", window_id, "-stats"], capture_output=True, text=True, timeout=2
            )

            if result.returncode != 0:
                return None

            geometry = {}
            for line in result.stdout.split("\n"):
                if "Absolute upper-left X:" in line:
                    geometry["x"] = int(line.split(":")[1].strip())
                elif "Absolute upper-left Y:" in line:
                    geometry["y"] = int(line.split(":")[1].strip())
                elif "Width:" in line:
                    geometry["width"] = int(line.split(":")[1].strip())
                elif "Height:" in line:
                    geometry["height"] = int(line.split(":")[1].strip())
                elif "Map State:" in line:
                    geometry["mapped"] = "IsViewable" in line

            return geometry if geometry else None

        except Exception as e:
            self.logger.debug(f"Failed to get geometry for {window_id}: {e}")
            return None

    def get_window_name(self, window_id: str) -> str:
        """Get window title with fallback"""
        if not self._available_tools.get("xdotool", False):
            self.logger.debug("xdotool not available for window name detection")
            return "Unknown"

        try:
            result = subprocess.run(
                ["xdotool", "getwindowname", window_id], capture_output=True, text=True, timeout=2
            )
            return result.stdout.strip() if result.returncode == 0 else "Unknown"
        except Exception:
            return "Unknown"

    def get_focused_window(self) -> Optional[str]:
        """Get currently focused window ID with fallback"""
        if not self._available_tools.get("xdotool", False):
            self.logger.debug("xdotool not available for focused window detection")
            return None

        try:
            result = subprocess.run(["xdotool", "getactivewindow"], capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                window_id = result.stdout.strip()
                return f"0x{int(window_id):x}" if window_id.isdigit() else None
            return None
        except Exception:
            return None

    def get_current_desktop(self) -> int:
        """Get current virtual desktop with fallback"""
        if not self._available_tools.get("xprop", False):
            self.logger.debug("xprop not available for desktop detection")
            return 0

        try:
            result = subprocess.run(
                ["xprop", "-root", "_NET_CURRENT_DESKTOP"], capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                match = re.search(r"= (\d+)", result.stdout)
                return int(match.group(1)) if match else 0
            return 0
        except Exception:
            return 0

    def get_window_desktop(self, window_id: str) -> int:
        """Get which desktop window is on with fallback"""
        if not self._available_tools.get("xprop", False):
            self.logger.debug("xprop not available for window desktop detection")
            return -1

        try:
            result = subprocess.run(
                ["xprop", "-id", window_id, "_NET_WM_DESKTOP"], capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                match = re.search(r"= (\d+)", result.stdout)
                return int(match.group(1)) if match else -1
            return -1
        except Exception:
            return -1

    def find_windows_for_app(self, app_name: str) -> List[str]:
        """Find all X11 window IDs for an application with fallback"""
        if not self._available_tools.get("xdotool", False):
            self.logger.debug("xdotool not available for app window search")
            return []

        try:
            # Try by class name
            result = subprocess.run(
                ["xdotool", "search", "--class", app_name], capture_output=True, text=True, timeout=2
            )

            if result.returncode == 0 and result.stdout.strip():
                window_ids = [f"0x{int(wid):x}" for wid in result.stdout.strip().split("\n") if wid]
                return window_ids

            # Fallback: try by name
            result = subprocess.run(
                ["xdotool", "search", "--name", app_name], capture_output=True, text=True, timeout=2
            )

            if result.returncode == 0 and result.stdout.strip():
                window_ids = [f"0x{int(wid):x}" for wid in result.stdout.strip().split("\n") if wid]
                return window_ids

            return []

        except Exception as e:
            self.logger.debug(f"Failed to find windows for {app_name}: {e}")
            return []

    def get_fallback_window_info(self) -> Dict[str, Any]:
        """Provide fallback window information when X11 tools are not available"""
        fallback_info = {
            "z_order": [],
            "focused_window": None,
            "current_desktop": 0,
            "available_tools": self._available_tools,
            "fallback_mode": True,
        }

        self.logger.info(f"Using fallback window management - available tools: {self._available_tools}")
        return fallback_info


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
        self.parser = HierarchicalParser()
        self.x11_wm = X11WindowManager()
        self.app_enhancer = AppEnhancer(controller)  # Pass controller to enhancer

        # Initialize autoglm_v tool classes
        self.tools = {
            "code": CodeTools,
            "chrome": BrowserTools,
            "vlc": VLCTools,
            "libreoffice_calc": CalcTools,
            "libreoffice_writer": WriterTools,
            "libreoffice_impress": ImpressTools,
        }

    def extract_app_states(self, accessibility_tree: str) -> List[AppState]:
        """Extract app states with X11 window manager integration and AT-SPI2 fallback"""
        try:
            # Step 1: Get real window z-order from X11 (if available)
            z_order_list = self.x11_wm.get_window_z_order()
            focused_window = self.x11_wm.get_focused_window()
            current_desktop = self.x11_wm.get_current_desktop()

            if z_order_list:
                self.logger.info(f"Found {len(z_order_list)} windows in X11 stacking order")
                self.logger.info(f"Focused window: {focused_window}")
            else:
                self.logger.info("X11 not available - using AT-SPI2 fallback mode")

            # Step 2: Parse AT-SPI2 tree
            root = ET.fromstring(accessibility_tree)

            # Step 3: Extract windows from AT-SPI2
            atspi_windows = self._extract_atspi_windows(root)

            if not atspi_windows:
                self.logger.warning("No windows found in AT-SPI2 tree")
                return []

            # Step 4: Match with X11 windows and assign real z-order (or use AT-SPI2 fallback)
            if z_order_list:
                window_states = self._match_x11_with_atspi(
                    atspi_windows, z_order_list, focused_window, current_desktop
                )
            else:
                window_states = self._create_atspi_fallback_windows(atspi_windows)

            # Step 5: Enhance with app-specific data (CDP, UNO)
            enhanced_window_states = self._enhance_with_app_specific_data(window_states)

            # Step 6: Convert to legacy AppState format
            app_states = []
            for window_state in enhanced_window_states:
                app_state = self._convert_window_to_app_state(window_state)
                if app_state:
                    app_states.append(app_state)

            self.logger.info(f"Extracted {len(app_states)} app states")
            return app_states

        except Exception as e:
            self.logger.error(f"Error extracting app states: {e}")
            return []

    def _extract_atspi_windows(self, root: ET.Element) -> List[Dict[str, Any]]:
        """Extract windows from AT-SPI2 tree"""
        windows = []

        for app_node in root:
            if app_node.tag != "application":
                continue

            app_name = app_node.get("name", "Unknown")

            # Skip system apps
            if self._should_skip_app(app_name):
                continue

            # Find windows/frames in this app
            for window_node in app_node:
                if window_node.tag not in ["frame", "window"]:
                    continue

                window_name = window_node.get("name", "Unnamed Window")

                # Parse elements in this window
                elements = self.parser._parse_element_tree(
                    window_node, parent_id=None, depth=0, parent_visibility=VisibilityState.VISIBLE
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

        for atspi_window in atspi_windows:
            # Find matching X11 window
            x11_window_id = self._find_matching_x11_window(
                atspi_window["app_name"], atspi_window["window_name"]
            )

            if not x11_window_id:
                self.logger.debug(f"No X11 window found for {atspi_window['window_name']}")
                continue

            # Get real geometry from X11
            geometry = self.x11_wm.get_window_geometry(x11_window_id)
            if not geometry:
                continue

            # Get z-order position
            try:
                z_order = len(z_order_list) - z_order_list.index(x11_window_id)
            except ValueError:
                z_order = 0

            # Get desktop
            window_desktop = self.x11_wm.get_window_desktop(x11_window_id)

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

            # Only include if visible on current desktop
            if window_desktop == -1 or window_desktop == current_desktop:
                if geometry.get("mapped", True):
                    window_states.append(window_state)

        # Sort by z-order (highest first = topmost)
        window_states.sort(key=lambda w: w.z_order, reverse=True)

        self.logger.info(f"Matched {len(window_states)} windows with X11 data")
        for w in window_states:
            self.logger.info(
                f"  z={w.z_order}: {w.window_name} ({w.app_name}) - {len(w.root_element.children) if w.root_element else 0} elements"
            )

        return window_states

    def _create_atspi_fallback_windows(self, atspi_windows: List[Dict[str, Any]]) -> List[WindowState]:
        """Create WindowState objects from AT-SPI2 data when X11 is not available"""
        window_states = []

        for i, atspi_window in enumerate(atspi_windows):
            # Create WindowState with AT-SPI2 data only
            window_state = WindowState(
                window_id=f"{atspi_window['app_name']}_{atspi_window['window_name']}",
                window_name=atspi_window["window_name"],
                app_name=atspi_window["app_name"],
                is_active=atspi_window.get("is_active", False),
                is_modal=atspi_window["is_modal"],
                geometry=atspi_window.get("geometry", {}),
                z_order=i,  # Simple ordering based on AT-SPI2 order
                x11_window_id=None,  # No X11 data available
                is_mapped=True,  # Assume mapped if in AT-SPI2
                desktop=0,  # Assume current desktop
                root_element=self._build_element_tree(atspi_window["elements"]),
            )

            window_states.append(window_state)

        # Sort by estimated z-order (active windows first)
        window_states.sort(key=lambda w: (w.is_active, w.is_modal), reverse=True)

        self.logger.info(f"Created {len(window_states)} windows using AT-SPI2 fallback")
        for w in window_states:
            self.logger.info(
                f"  {w.window_name} ({w.app_name}) - {len(w.root_element.children) if w.root_element else 0} elements"
            )

        return window_states

    def _find_matching_x11_window(self, app_name: str, window_name: str) -> Optional[str]:
        """Find X11 window ID that matches AT-SPI2 window"""

        # Get candidate windows from X11
        window_ids = self.x11_wm.find_windows_for_app(app_name)

        # Match by window title
        for window_id in window_ids:
            x11_title = self.x11_wm.get_window_name(window_id)

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
        """Build element tree from flat list"""
        if not elements:
            return None

        # Find root element (no parent)
        root_elements = [elem for elem in elements if elem.parent_id is None]
        if not root_elements:
            return elements[0]  # Fallback to first element

        root = root_elements[0]

        # Build parent-child relationships
        element_map = {elem.element_id: elem for elem in elements}

        for elem in elements:
            if elem.parent_id and elem.parent_id in element_map:
                parent = element_map[elem.parent_id]
                parent.children.append(elem)

        return root

    def _enhance_with_app_specific_data(self, window_states: List[WindowState]) -> List[WindowState]:
        """Enhance window states with app-specific data (CDP, UNO)"""
        enhanced_states = []

        for window_state in window_states:
            app_type = self._map_app_name_to_type(window_state.app_name)

            # Get base elements
            base_elements = window_state.get_all_elements(include_structural=False)

            # Enhance with app-specific data
            enhanced_elements = self._enhance_elements_for_app(app_type, window_state.app_name, base_elements)

            # Create enhanced window state
            enhanced_window_state = WindowState(
                window_id=window_state.window_id,
                window_name=window_state.window_name,
                app_name=window_state.app_name,
                is_active=window_state.is_active,
                is_modal=window_state.is_modal,
                is_minimized=window_state.is_minimized,
                geometry=window_state.geometry,
                z_order=window_state.z_order,
                x11_window_id=window_state.x11_window_id,
                is_mapped=window_state.is_mapped,
                desktop=window_state.desktop,
                root_element=self._build_element_tree_from_flat(enhanced_elements),
            )

            enhanced_states.append(enhanced_window_state)

        return enhanced_states

    def _enhance_elements_for_app(
        self, app_type: str, app_name: str, base_elements: List[UIElement]
    ) -> List[UIElement]:
        """Enhance AT-SPI2 elements with app-specific data"""

        # Use simplified enhancer (placeholder for future CDP/UNO integration)
        if app_type == "chrome":
            return self.app_enhancer.enhance_chrome_elements(base_elements)
        elif app_type == "code":
            return self.app_enhancer.enhance_vscode_elements(base_elements)
        elif app_type.startswith("libreoffice"):
            return self.app_enhancer.enhance_libreoffice_elements(base_elements, app_type)
        else:
            return base_elements

    def _build_element_tree_from_flat(self, elements: List[UIElement]) -> Optional[UIElement]:
        """Build element tree from flat list"""
        if not elements:
            return None

        # Find root element (no parent)
        root_elements = [elem for elem in elements if elem.parent_id is None]
        if not root_elements:
            return elements[0]  # Fallback to first element

        root = root_elements[0]

        # Build parent-child relationships
        element_map = {elem.element_id: elem for elem in elements}

        for elem in elements:
            if elem.parent_id and elem.parent_id in element_map:
                parent = element_map[elem.parent_id]
                parent.children.append(elem)

        return root

    def _should_skip_app(self, app_name: str) -> bool:
        """Skip system/background apps"""
        skip_apps = ["vmware-user", "gsd-", "ibus-", "evolution-alarm", "xdg-desktop-portal"]
        return any(skip in app_name for skip in skip_apps)

    def _convert_window_to_app_state(self, window_state: WindowState) -> Optional[AppState]:
        """Convert WindowState to legacy AppState format"""
        try:
            # Get all visible elements
            ui_elements = window_state.get_all_elements(include_structural=False)

            # Convert UIElement to AppElement
            app_elements = []
            for ui_elem in ui_elements:
                app_element = AppElement(
                    element_id=ui_elem.element_id,
                    element_type=ui_elem.element_type,
                    name=ui_elem.name,
                    text=ui_elem.name,
                    position=ui_elem.position,
                    properties=ui_elem.properties,
                )
                app_elements.append(app_element)

            if not app_elements:
                return None

            # Determine app type
            app_type = self._map_app_name_to_type(window_state.app_name)
            properties = self._get_app_properties(app_type)

            # Add window-specific properties
            properties.update(
                {
                    "window_title": window_state.window_name,
                    "window_z_order": window_state.z_order,
                    "is_top_window": window_state.is_active,
                    "is_modal": window_state.is_modal,
                    "geometry": window_state.geometry,
                    "x11_window_id": window_state.x11_window_id,
                    "is_mapped": window_state.is_mapped,
                    "desktop": window_state.desktop,
                }
            )

            return AppState(
                app_name=window_state.app_name,
                app_type=app_type,
                window_title=window_state.window_name,
                elements=app_elements,
                properties=properties,
                timestamp=self._get_timestamp(),
            )

        except Exception as e:
            self.logger.error(f"Error converting window to app state: {e}")
            return None

    def _map_app_name_to_type(self, app_name: str) -> str:
        """Map application name to autoglm_v tool type"""
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
            # For LibreOffice, we need to check window name to determine specific app
            return "libreoffice"  # Will be refined by window name
        elif "vlc" in app_name_lower or "media" in app_name_lower:
            return "vlc"
        elif "gnome-shell" in app_name_lower:
            return "desktop"
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


class HierarchicalParser:
    """Parses accessibility tree into hierarchical structure"""

    def __init__(self, platform: str = "Ubuntu"):
        self.platform = platform
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
        self, node: ET.Element, parent_id: Optional[str], depth: int, parent_visibility: VisibilityState
    ) -> Optional[UIElement]:
        """Recursively parse element and its children"""

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
        if visibility != VisibilityState.HIDDEN_COLLAPSED:
            for child_node in node:
                child_element = self._parse_element_tree(
                    child_node, parent_id=element.element_id, depth=depth + 1, parent_visibility=visibility
                )
                if child_element:
                    element.children.append(child_element)

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


class ElementTester:
    """Simple element validity testing with coordinate-based validation"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def test_element_validity_simple(self, element: AppElement, action_str: str, env) -> Dict[str, Any]:
        """
        Simple element validity testing using coordinate-based validation and single-step execution.

        This addresses the main failure cases with simpler, more maintainable code:
        1. Wrong element → No state change
        2. Wrong element → Wrong state change (detected by coordinate validation)

        Returns:
            Dict with 'valid': bool, 'reason': str, 'score': float
        """
        try:
            # Coordinate-based validation (most important)
            coord_score = self._validate_element_coordinates(element)
            if coord_score < 0.5:
                return {
                    "valid": False,
                    "reason": f"Invalid coordinates (score: {coord_score:.2f})",
                    "score": coord_score,
                }

            # Element type validation
            type_score = self._validate_element_type(element)

            # Size and position validation
            size_score = self._validate_element_size(element)

            # Combined score
            total_score = (coord_score * 0.5) + (type_score * 0.3) + (size_score * 0.2)

            # Determine validity
            is_valid = total_score >= 0.6

            reason = f"Score: {total_score:.2f} (coord: {coord_score:.2f}, type: {type_score:.2f}, size: {size_score:.2f})"

            return {
                "valid": is_valid,
                "reason": reason,
                "score": total_score,
                "coord_score": coord_score,
                "type_score": type_score,
                "size_score": size_score,
            }

        except Exception as e:
            self.logger.error(f"Error testing element validity: {e}")
            return {
                "valid": False,
                "reason": f"Test execution failed: {str(e)}",
                "score": 0.0,
                "error": str(e),
            }

    def _validate_element_coordinates(self, element: AppElement) -> float:
        """Validate element coordinates - most important for detecting invisible elements"""
        center_x = element.position.get("center_x", 0)
        center_y = element.position.get("center_y", 0)
        width = element.position.get("width", 0)
        height = element.position.get("height", 0)

        score = 1.0

        # Check screen bounds
        if center_x < 0 or center_y < 0 or center_x > 1920 or center_y > 1080:
            return 0.0

        # Check reasonable bounds (not at screen edges)
        if center_x < 50 or center_x > 1870 or center_y < 50 or center_y > 1030:
            score -= 0.3

        # Check for suspicious coordinates (like 0,0 which often indicates hidden elements)
        if center_x == 0 or center_y == 0:
            score -= 0.5

        # Check size reasonableness
        if width < 8 or height < 8:
            score -= 0.4
        elif width > 500 or height > 500:
            score -= 0.2

        return max(0.0, score)

    def _validate_element_type(self, element: AppElement) -> float:
        """Validate element type appropriateness"""
        element_type = element.element_type.lower()

        # High priority interactive types
        high_priority = ["button", "check-box", "radio-button", "combo-box", "push-button", "tab"]
        if element_type in high_priority:
            return 1.0

        # Medium priority types
        medium_priority = ["menu-item", "list-item", "textfield", "textarea"]
        if element_type in medium_priority:
            return 0.7

        # Lower priority types
        low_priority = ["label", "text", "static"]
        if element_type in low_priority:
            return 0.3

        # Unknown types
        return 0.5

    def _validate_element_size(self, element: AppElement) -> float:
        """Validate element size and aspect ratio"""
        width = element.position.get("width", 0)
        height = element.position.get("height", 0)

        if width <= 0 or height <= 0:
            return 0.0

        score = 1.0

        # Check aspect ratio (very skewed elements are likely containers)
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > 10:
            score -= 0.5
        elif aspect_ratio > 5:
            score -= 0.2

        # Check area (very small or very large elements)
        area = width * height
        if area < 64:  # Too small
            score -= 0.3
        elif area > 100000:  # Too large
            score -= 0.2

        return max(0.0, score)

    def _update_action_coordinates(self, action_str: str, position: Dict[str, int]) -> str:
        """Update action coordinates to the exact center of the element position"""
        import re

        new_x = position["center_x"]
        new_y = position["center_y"]

        coord_pattern = r"(\d+),\s*(\d+)"

        def replace_coords(match):
            return f"{new_x}, {new_y}"

        updated_action = re.sub(coord_pattern, replace_coords, action_str)
        return updated_action


class AutoglmElementTracker:
    """Track UI elements using autoglm_v tools"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.grounding_agent = GroundingAgent()
        from perturbation_engine.pipeline.clean_llm_services import CleanElementIdentificationLLM

        self.llm = CleanElementIdentificationLLM()

    def identify_target_element_candidates(
        self, action_str: str, app_states: List[AppState]
    ) -> List[AppElement]:
        """
        Identify ALL possible target element candidates using LLM-based approach.

        Returns a list of AppElement candidates ranked by likelihood for multi-rollout testing.
        """
        try:
            # Convert AppState objects to dictionaries for LLM processing
            app_states_dict = [
                app_state.to_dict() if hasattr(app_state, "to_dict") else app_state
                for app_state in app_states
            ]

            # Use LLM to identify ALL possible target elements
            llm_candidates = self._identify_candidates_with_llm(action_str, app_states_dict)
            if not llm_candidates:
                self.logger.warning(f"✗ No target element candidates found for: {action_str[:100]}")
                return []

            # Convert LLM candidates to actual AppElement objects
            element_candidates = []
            for llm_candidate in llm_candidates:
                target_element = self._find_element_by_identifier(llm_candidate, app_states)
                if target_element:
                    element_candidates.append(target_element)

            self.logger.info(
                f"✓ Found {len(element_candidates)} target element candidates for: {action_str[:50]}..."
            )
            return element_candidates

        except Exception as e:
            self.logger.error(f"Error identifying target element candidates: {e}")
            return []

    def identify_target_element(self, action_str: str, app_states: List[AppState]) -> Optional[AppElement]:
        """
        Identify single target element (backward compatibility).

        Returns the first (highest confidence) candidate from identify_target_element_candidates.
        """
        candidates = self.identify_target_element_candidates(action_str, app_states)
        return candidates[0] if candidates else None

    def _identify_candidates_with_llm(
        self, action_str: str, app_states: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Use LLM to identify ALL possible target element candidates"""
        try:
            retries = 0
            while retries < 3:
                retries += 1
                result = self.llm.identify_target_element_candidates(action_str, app_states)
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
