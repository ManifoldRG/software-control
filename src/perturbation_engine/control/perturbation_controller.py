"""
PerturbationController: Execute perturbation code
Clean interface for VM manipulation
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from playwright.sync_api import Page, sync_playwright

from OSWorld.desktop_env.controllers.python import PythonController
from OSWorld.desktop_env.controllers.setup import SetupController
from perturbation_engine.control.app_state_extractor import AppStateExtractor


@dataclass
class ManipulationResult:
    """Result of VM manipulation operation"""

    success: bool
    operation_type: str
    target_app: str
    result_data: Dict[str, Any]
    error_message: Optional[str] = None


class PerturbationController(PythonController, SetupController):
    """Execute perturbation code with clean interface"""

    def __init__(self, vm_ip: str, server_port: int, chromium_port: int = 9222, **kwargs):
        # Ensure logging is configured for subprocess (only if not already configured)
        if not logging.getLogger().handlers:
            from perturbation_engine.configure_logging import configure_logging

            configure_logging()

        PythonController.__init__(self, vm_ip, server_port, **kwargs)
        SetupController.__init__(self, vm_ip, server_port, chromium_port, **kwargs)
        self.vm_ip = vm_ip
        self.server_port = server_port
        self.chromium_port = chromium_port
        self.logger = logging.getLogger(__name__)

        # Playwright connection
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

        # App state extractor
        self._app_state_extractor = None
        if AppStateExtractor:
            self._app_state_extractor = AppStateExtractor(self)

        # Coordinate tracking for ground truth action updates
        self._element_position_tracker = {}

    def execute_perturbation(
        self, perturbation_type: str, generated_code: str, api_call: str, parameters: Dict[str, Any]
    ) -> ManipulationResult:
        """
        Execute perturbation using generated code with sophisticated handling.

        Tracks UI element positions before/after layout-changing perturbations
        to enable ground truth action coordinate updates for visual invariance learning.
        """
        try:
            # Capture element positions BEFORE perturbation
            pre_perturbation_positions = self._capture_element_positions()

            success = False
            result_data = {}

            if api_call == "execute_js_on_page":
                success = self.execute_js_on_page(generated_code)
                result_data = {"api_call": api_call, "code": generated_code}
            elif api_call == "execute_bash_command":
                success = self.execute_bash_command(generated_code)
                result_data = {"api_call": api_call, "command": generated_code}
            elif api_call == "execute_python_command":
                result = self.execute_python_command(generated_code)
                success = result.get("status") == "success"
                result_data = {"api_call": api_call, "result": result}
            elif api_call == "execute_uno_command":
                success = self.execute_uno_command(generated_code, parameters)
                result_data = {"api_call": api_call, "code": generated_code}
            elif api_call == "manipulate_app_state":
                success = self._manipulate_app_state(parameters)
                result_data = {"api_call": api_call, "parameters": parameters}
            elif api_call == "execute_system_perturbation":
                # Handle sophisticated system-level perturbations
                system_type = parameters.get("system_type", "desktop_theme")
                success = self.execute_system_perturbation(system_type, parameters)
                result_data = {"api_call": api_call, "system_type": system_type, "parameters": parameters}
            else:
                self.logger.warning(f"Unknown API call: {api_call}")
                success = False
                result_data = {"api_call": api_call, "error": "Unknown API call"}

            # Capture element positions AFTER perturbation
            post_perturbation_positions = self._capture_element_positions()

            # Calculate position deltas for coordinate tracking
            position_changes = self._calculate_position_changes(
                pre_perturbation_positions, post_perturbation_positions
            )

            # Store for later retrieval by trajectory generator
            if position_changes:
                result_data["position_changes"] = position_changes
                self.logger.info(f"Detected {len(position_changes)} element position changes")

            return ManipulationResult(
                success=success,
                operation_type=perturbation_type,
                target_app=parameters.get("target_app", "unknown"),
                result_data=result_data,
                error_message=None if success else f"Failed to execute {api_call}",
            )

        except Exception as e:
            self.logger.error(f"Error executing perturbation: {e}")
            return ManipulationResult(
                success=False,
                operation_type=perturbation_type,
                target_app=parameters.get("target_app", "unknown"),
                result_data={"error": str(e)},
                error_message=str(e),
            )

    def execute_js_on_page(self, js_code: str) -> bool:
        """Execute JavaScript code on the current page"""
        try:
            page = self._get_page()
            if not page:
                return False

            # Clean up the JavaScript code
            if "```" in js_code:
                js_code = js_code.split("```")[1].removeprefix("javascript").strip()

            page.evaluate(js_code)
            self.logger.info(f"Executed JavaScript: {js_code[:100]}...")
            return True

        except Exception as e:
            self.logger.error(f"Error executing JavaScript: {e}")
            return False

    def execute_bash_command(self, command: str) -> bool:
        """
        Execute bash command with improved error handling.

        Now uses run_bash_script() to properly handle shell special characters
        like pipes, redirects, conditionals, and background processes.

        Checks BOTH status=="success" AND returncode==0 for true success.
        """
        try:
            # Clean up the command if it contains markdown
            if "```" in command:
                command = command.split("```")[1].removeprefix("bash").strip()

            # Use run_bash_script for proper shell handling
            result = self.run_bash_script(command, timeout=30)

            # Check both status and return code
            # python.py sometimes returns status="success" even with errors
            if result and result.get("status") == "success" and result.get("returncode", -1) == 0:
                self.logger.info(f"Bash command executed successfully: {command}")
                return True
            else:
                self.logger.warning(f"Bash command failed: {command}")
                if result:
                    self.logger.warning(
                        f"Status: {result.get('status')}, Return code: {result.get('returncode')}, Error: {result.get('error', '')}"
                    )
                return False

        except Exception as e:
            self.logger.error(f"Error executing bash command: {e}")
            return False

    def execute_python_command(self, python_code: str) -> Dict[str, Any]:
        """Execute Python code"""
        try:
            result = super().execute_python_command(python_code)
            # Ensure the result has the correct structure
            if "status" not in result:
                if result.get("success", False):
                    result["status"] = "success"
                else:
                    result["status"] = "error"
            return result
        except Exception as e:
            self.logger.error(f"Error executing Python: {e}")
            return {"status": "error", "error": str(e)}

    def execute_uno_command(self, uno_code: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute UNO command for LibreOffice manipulation.

        Returns:
            Dict with status, output, and error information
        """
        try:
            # Clean up the UNO code
            if "```" in uno_code:
                uno_code = uno_code.split("```")[1].removeprefix("python").strip()

            # Indent the UNO code for insertion into try block
            indented_uno_code = "\n".join("    " + line for line in uno_code.split("\n"))

            # Execute UNO code via Python with robust LibreOffice connection
            python_wrapper = f"""
import uno
import unohelper
import subprocess
import time
from com.sun.star.uno import RuntimeException

def identify_document_type(component):
    if component.supportsService("com.sun.star.sheet.SpreadsheetDocument"):
        return "Calc"
    if component.supportsService("com.sun.star.text.TextDocument"):
        return "Writer"
    if component.supportsService("com.sun.star.sheet.PresentationDocument"):
        return "Impress"
    return None

try:
    # Clean up previous TCP connections
    subprocess.run(
        'echo "osworld-public-evaluation" | sudo -S ss --kill --tcp state TIME-WAIT sport = :2002',
        shell=True,
        check=False,
        text=True,
        capture_output=True
    )

    # Start LibreOffice headless
    soffice_process = subprocess.Popen([
        "soffice",
        "--headless",
        "--invisible",
        "--accept=socket,host=localhost,port=2002;urp;StarOffice.Service"
    ])

    # Wait for LibreOffice to start
    time.sleep(3)

    # Get LibreOffice context
    localContext = uno.getComponentContext()
    resolver = localContext.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver", localContext
    )
    context = resolver.resolve(
        "uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext"
    )
    desktop = context.ServiceManager.createInstanceWithContext(
        "com.sun.star.frame.Desktop", context
    )

    # Execute the UNO code
{indented_uno_code}

    print("UNO command executed successfully")

    # Clean up
    soffice_process.terminate()
    soffice_process.wait()

except Exception as e:
    print(f"UNO command failed: {{e}}")
    try:
        soffice_process.terminate()
        soffice_process.wait()
    except:
        pass
"""

            result = self.execute_python_command(python_wrapper)
            return result

        except Exception as e:
            self.logger.error(f"Error executing UNO command: {e}")
            return {"status": "error", "error": str(e)}

    def _capture_element_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Capture current positions of all interactive UI elements.

        Returns dict mapping element identifiers to their positions:
        {
            "button_Save": {"x": 100, "y": 200, "width": 80, "height": 30},
            "link_Home": {"x": 150, "y": 50, "width": 60, "height": 20},
            ...
        }
        """
        try:
            app_states = self.get_comprehensive_app_states(use_comprehensive=False)
            if not app_states:
                return {}

            positions = {}

            for app_state in app_states:
                # Extract positions from categorized elements
                for category in [
                    "buttons",
                    "links",
                    "text_fields",
                    "menu_items",
                    "checkboxes",
                    "radio_buttons",
                    "combo_boxes",
                    "tabs",
                    "images",
                ]:
                    elements = app_state.get(category, [])
                    for elem in elements:
                        position = elem.get("position", {})
                        if position and position.get("center_x") and position.get("center_y"):
                            # Create unique identifier from element properties
                            elem_id = self._create_element_id(elem, category)
                            positions[elem_id] = {
                                "center_x": position["center_x"],
                                "center_y": position["center_y"],
                                "x": position.get("x", 0),
                                "y": position.get("y", 0),
                                "width": position.get("width", 0),
                                "height": position.get("height", 0),
                                "category": category,
                                "name": elem.get("name", ""),
                                "text": elem.get("text", ""),
                            }

            return positions

        except Exception as e:
            self.logger.warning(f"Error capturing element positions: {e}")
            return {}

    def _create_element_id(self, elem: Dict[str, Any], category: str) -> str:
        """
        Create a unique, stable identifier for an element.

        Uses a combination of category, role, name, and text to identify elements
        even after perturbations that might change their internal IDs.
        """
        name = elem.get("name", "")
        text = elem.get("text", "")[:50]  # Truncate for stability

        # Clean and normalize
        name = name.strip().replace(" ", "_")
        text = text.strip().replace(" ", "_")

        if name:
            return f"{category}_{name}"
        elif text:
            return f"{category}_{text}"
        else:
            # Fallback to position-based ID (less stable but unique)
            pos = elem.get("position", {})
            x = pos.get("center_x", 0)
            y = pos.get("center_y", 0)
            return f"{category}_at_{x}_{y}"

    def _calculate_position_changes(
        self, before: Dict[str, Dict[str, Any]], after: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, int]]:
        """
        Calculate position deltas for elements that moved.

        Returns:
            Dict mapping element IDs to position deltas:
            {
                "button_Save": {"dx": 50, "dy": 100, "old_x": 100, "old_y": 200, "new_x": 150, "new_y": 300},
                ...
            }
        """
        changes = {}

        for elem_id, before_pos in before.items():
            if elem_id in after:
                after_pos = after[elem_id]

                dx = after_pos["center_x"] - before_pos["center_x"]
                dy = after_pos["center_y"] - before_pos["center_y"]

                # Only track significant movements (> 5 pixels)
                if abs(dx) > 5 or abs(dy) > 5:
                    changes[elem_id] = {
                        "dx": dx,
                        "dy": dy,
                        "old_center_x": before_pos["center_x"],
                        "old_center_y": before_pos["center_y"],
                        "new_center_x": after_pos["center_x"],
                        "new_center_y": after_pos["center_y"],
                        "category": before_pos["category"],
                        "name": before_pos["name"],
                        "text": before_pos["text"],
                    }

        return changes

    def update_action_coordinates(self, action: str, position_changes: Dict[str, Dict[str, int]]) -> str:
        """
        Update action coordinates based on element position changes.

        Parses pyautogui.click(x, y) and matches coordinates to moved elements,
        then updates with new coordinates.

        Args:
            action: Original action string (e.g., "pyautogui.click(100, 200, duration=1)")
            position_changes: Dict of element position deltas from perturbation

        Returns:
            Updated action string with corrected coordinates
        """
        import re

        # Extract coordinates from pyautogui calls
        click_pattern = r"pyautogui\.(click|rightClick|doubleClick|moveTo)\((\d+),\s*(\d+)"
        match = re.search(click_pattern, action)

        if not match:
            # No coordinates to update
            return action

        # method = match.group(1)
        old_x = int(match.group(2))
        old_y = int(match.group(3))

        # Find the element that matches these coordinates
        best_match = None
        min_distance = float("inf")

        for _, change in position_changes.items():
            # Calculate distance from action coordinates to element's OLD position
            distance = ((old_x - change["old_center_x"]) ** 2 + (old_y - change["old_center_y"]) ** 2) ** 0.5

            # If action coordinates are close to an element's old position, it's likely the target
            if distance < min_distance and distance < 50:  # Within 50 pixels
                min_distance = distance
                best_match = change

        if best_match:
            # Update coordinates to element's new position
            new_x = best_match["new_center_x"]
            new_y = best_match["new_center_y"]

            # Replace coordinates in action string
            old_coords = f"{old_x}, {old_y}"
            new_coords = f"{new_x}, {new_y}"
            updated_action = action.replace(old_coords, new_coords)

            self.logger.info(
                f"Updated action coordinates: {old_coords} -> {new_coords} "
                f"(element: {best_match['name'] or best_match['text'][:20]})"
            )

            return updated_action

        # No matching element found, return original
        return action

    def _manipulate_app_state(self, parameters: Dict[str, Any]) -> bool:
        """Manipulate app state based on parameters"""
        try:
            app_type = parameters.get("target_app", "unknown")
            operation = parameters.get("operation", "unknown")

            if operation == "switch_to_app":
                return self._switch_to_app(app_type)
            elif operation == "resize_window":
                return self._resize_window(app_type, parameters)
            elif operation == "close_app":
                return self._close_app(app_type)
            else:
                self.logger.warning(f"Unknown app manipulation: {operation}")
                return False

        except Exception as e:
            self.logger.error(f"Error manipulating app state: {e}")
            return False

    def _switch_to_app(self, app_name: str) -> bool:
        """Switch to specific application"""
        try:
            # Use wmctrl to switch to app
            result = self.execute_python_command(
                f"import subprocess; subprocess.run(['wmctrl', '-a', '{app_name}'])"
            )
            return result.get("status") == "success"
        except Exception as e:
            self.logger.error(f"Error switching to app {app_name}: {e}")
            return False

    def _resize_window(self, app_name: str, parameters: Dict[str, Any]) -> bool:
        """Resize application window"""
        try:
            width = parameters.get("width", 1920)
            height = parameters.get("height", 1080)
            result = self.execute_python_command(
                f"import subprocess; subprocess.run(['wmctrl', '-r', '{app_name}', '-e', '0,0,0,{width},{height}'])"
            )
            return result.get("status") == "success"
        except Exception as e:
            self.logger.error(f"Error resizing window for {app_name}: {e}")
            return False

    def _close_app(self, app_name: str) -> bool:
        """Close application"""
        try:
            result = self.execute_python_command(
                f"import subprocess; subprocess.run(['pkill', '-f', '{app_name}'])"
            )
            return result.get("status") == "success"
        except Exception as e:
            self.logger.error(f"Error closing app {app_name}: {e}")
            return False

    def execute_system_perturbation(self, perturbation_type: str, parameters: Dict[str, Any]) -> bool:
        """Execute sophisticated system-level perturbations"""
        try:
            if perturbation_type == "desktop_theme":
                theme = parameters.get("theme", "Adwaita-dark")
                icon_theme = parameters.get("icon_theme", "Papirus-Dark")
                commands = [
                    f"gsettings set org.gnome.desktop.interface gtk-theme '{theme}'",
                    f"gsettings set org.gnome.desktop.interface icon-theme '{icon_theme}'",
                ]
                for cmd in commands:
                    self.execute_bash_command(cmd)
                return True

            elif perturbation_type == "desktop_wallpaper":
                wallpaper = parameters.get("wallpaper", "/usr/share/backgrounds/gnome/adwaita-morning.jpg")
                self.execute_bash_command(
                    f"gsettings set org.gnome.desktop.background picture-uri 'file://{wallpaper}'"
                )
                return True

            elif perturbation_type == "system_notification":
                title = parameters.get("title", "Background Process")
                message = parameters.get("message", "System update running")
                self.execute_bash_command(f"notify-send '{title}' '{message}'")
                return True

            elif perturbation_type == "background_files":
                base_dir = parameters.get("base_dir", "/tmp/background_work")
                task_id = parameters.get("task_id", "unknown")
                self.execute_bash_command(
                    f"mkdir -p {base_dir}/{task_id} && touch {base_dir}/{task_id}/process.log"
                )
                return True

            elif perturbation_type == "window_management":
                app_name = parameters.get("app_name", "Calculator")
                x = parameters.get("x", 100)
                y = parameters.get("y", 100)
                width = parameters.get("width", 300)
                height = parameters.get("height", 200)
                self.execute_bash_command(f"wmctrl -r '{app_name}' -e 0,{x},{y},{width},{height}")
                return True

            else:
                self.logger.warning(f"Unknown system perturbation type: {perturbation_type}")
                return False

        except Exception as e:
            self.logger.error(f"Error executing system perturbation: {e}")
            return False

    def _get_page(self) -> Optional[Page]:
        """Get Playwright page with connection management"""
        if self._page is not None:
            return self._page

        try:
            self._playwright = sync_playwright().start()
            remote_debugging_url = f"http://{self.vm_ip}:{self.chromium_port}"

            # Connect to existing Chrome instance
            self._browser = self._playwright.chromium.connect_over_cdp(remote_debugging_url)

            # Get the first context and page
            if self._browser.contexts:
                self._context = self._browser.contexts[0]
                if self._context.pages:
                    self._page = self._context.pages[0]
                else:
                    self._page = self._context.new_page()
            else:
                self._context = self._browser.new_context()
                self._page = self._context.new_page()

            self.logger.info(f"Connected to Chrome via Playwright at {remote_debugging_url}")
            return self._page

        except Exception as e:
            self.logger.error(f"Failed to connect to Chrome via Playwright: {e}")
            return None

    def close_playwright(self):
        """Close Playwright connections"""
        try:
            if self._playwright:
                self._playwright.stop()
                self._playwright = None
                self._browser = None
                self._context = None
                self._page = None
                self.logger.info("Playwright connections closed")
        except Exception as e:
            self.logger.error(f"Error closing Playwright: {e}")

    def ensure_accessibility_enabled(self) -> bool:
        """
        Ensure AT-SPI accessibility is enabled for all applications.

        Uses execute_python_command instead of run_bash_script to avoid
        the _append_event bug in old VM code.

        Quick, non-blocking setup that won't hang the environment initialization.

        Returns:
            True if setup successful
        """
        try:
            self.logger.info("Setting up AT-SPI accessibility...")

            # Quick Python setup - minimal wait time
            python_code = """
import subprocess
import time

try:
    # Enable accessibility via gsettings (quick, doesn't hang)
    subprocess.run(['gsettings', 'set', 'org.gnome.desktop.interface', 'toolkit-accessibility', 'true'],
                   capture_output=True, text=True, timeout=2)
    subprocess.run(['gsettings', 'set', 'org.gnome.desktop.interface', 'accessibility', 'true'],
                   capture_output=True, text=True, timeout=2)

    # Check if AT-SPI bus is running
    result = subprocess.run(['pgrep', '-x', 'at-spi-bus-launcher'],
                           capture_output=True, text=True, timeout=1)

    if result.returncode != 0:
        # Launch in background (fire and forget - don't wait)
        subprocess.Popen(['/usr/libexec/at-spi-bus-launcher', '--launch-immediately'],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print('AT-SPI bus launched')
    else:
        print('AT-SPI bus already active')

except Exception as e:
    print(f'AT-SPI setup warning: {e}')
"""

            # Execute with Python (avoids bash script endpoint bug)
            # Use shorter timeout to avoid hanging
            result = self.execute_python_command(python_code)

            if result and result.get("status") == "success":
                output = result.get("output", "").strip()
                self.logger.info(f"AT-SPI setup result: {output}")
                return True
            else:
                self.logger.warning(f"AT-SPI setup returned non-success: {result}")
                # Return True anyway - accessibility might still work
                return True

        except Exception as e:
            self.logger.error(f"Error setting up accessibility (non-fatal): {e}")
            # Return True to avoid blocking - accessibility might already be enabled
            return True

    def get_comprehensive_app_states(self, use_comprehensive: bool = True) -> list:
        """
        Get app state information for LLM consumption.

        Args:
            use_comprehensive: If True, extract rich DOM/UNO data (slower but detailed)
                             If False, use basic accessibility tree only (faster)

        Comprehensive mode returns:
        - Browser: Full DOM (buttons, links, forms, inputs, headings)
        - LibreOffice: Document state (sheets, content, slides)
        - All apps: Categorized elements (buttons, menus, text fields, etc.)
        - UI structure (menu bars, toolbars, dialogs, panels)
        - Interactive elements with full metadata

        Basic mode returns:
        - app_type, app_name, current_view
        - key_elements (up to 10 interactive elements)
        - element_count
        """
        if not self._app_state_extractor:
            self.logger.warning("AppStateExtractor not available")
            return []

        try:
            return self._app_state_extractor.extract_app_states(use_comprehensive=use_comprehensive)
        except Exception as e:
            self.logger.error(f"Error extracting app states: {e}")
            # Try basic mode as fallback if comprehensive fails
            if use_comprehensive:
                self.logger.info("Falling back to basic extraction")
                try:
                    return self._app_state_extractor.extract_app_states(use_comprehensive=False)
                except Exception as e2:
                    self.logger.error(f"Basic extraction also failed: {e2}")
            return []
