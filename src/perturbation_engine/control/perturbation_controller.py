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
from perturbation_engine.control.clean_app_state_extractor import CleanAppStateExtractor
from perturbation_engine.tools.autoglm_integration import AutoglmAppStateExtractor


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

        self._app_state_extractor = CleanAppStateExtractor(self)
        self._autoglm_extractor = AutoglmAppStateExtractor()

    def execute_perturbation(
        self, perturbation_type: str, generated_code: str, api_call: str, parameters: Dict[str, Any]
    ) -> ManipulationResult:
        """
        Execute perturbation using generated code with sophisticated handling.

        Note: Coordinate tracking is now handled externally by trajectory_generator
        which identifies the target element from ground truth action coordinates,
        then updates those coordinates after perturbation if the element moved.
        """
        try:
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
                result = self.execute_uno_command(generated_code, parameters)
                success = result.get("status") == "success" and result.get("returncode", -1) == 0
                result_data = {"api_call": api_call, "code": generated_code, "result": result}
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

            # Extract detailed error message if available
            error_message = None
            if not success:
                if api_call == "execute_uno_command" and "result" in result_data:
                    result = result_data["result"]
                    if result.get("error"):
                        error_message = f"UNO command failed: {result['error']}"
                    elif result.get("returncode", 0) != 0:
                        error_message = f"UNO command failed with return code {result.get('returncode', -1)}"
                    else:
                        error_message = f"UNO command failed: {result.get('output', 'Unknown error')}"
                elif api_call == "execute_python_command" and "result" in result_data:
                    result = result_data["result"]
                    if result.get("error"):
                        error_message = f"Python command failed: {result['error']}"
                    else:
                        error_message = f"Python command failed: {result.get('output', 'Unknown error')}"
                else:
                    error_message = f"Failed to execute {api_call}"

            return ManipulationResult(
                success=success,
                operation_type=perturbation_type,
                target_app=parameters.get("target_app", "unknown"),
                result_data=result_data,
                error_message=error_message,
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
        Execute UNO command by sending it to the VM server for execution.
        The VM server handles LibreOffice process management and UNO API calls.
        """
        try:
            # Format UNO code to ensure proper structure
            formatted_uno_code = self._format_python_code(uno_code)

            # Build Python wrapper that handles LibreOffice process management on the VM
            python_wrapper = self._build_uno_python_wrapper(formatted_uno_code)

            # Validate the complete wrapper has valid syntax
            if not self._validate_python_syntax(python_wrapper):
                self.logger.error(f"UNO wrapper has syntax errors. Code snippet: {uno_code[:200]}")
                return {"status": "error", "error": "UNO wrapper syntax error", "returncode": 1}

            # Send to VM server for execution
            result = self.execute_python_command(python_wrapper)

            # Validate result
            if result and result.get("status") == "success":
                return result
            else:
                self.logger.warning(f"UNO execution failed: {result}")
                return result or {"status": "error", "error": "No result from VM", "returncode": 1}

        except Exception as e:
            self.logger.error(f"UNO execution error: {e}")
            return {"status": "error", "error": str(e), "returncode": 1}

    def _build_uno_python_wrapper(self, formatted_uno_code: str) -> str:
        """
        Build the Python wrapper for UNO execution with robust error handling.
        """
        # Ensure code is properly indented for the try block
        indented_code = "\n".join(
            "    " + line if line.strip() else "" for line in formatted_uno_code.split("\n")
        )

        return f"""import uno
import subprocess
import time
from com.sun.star.uno import RuntimeException

def identify_document_type(component):
    if component.supportsService("com.sun.star.sheet.SpreadsheetDocument"):
        return "Calc"
    if component.supportsService("com.sun.star.text.TextDocument"):
        return "Writer"
    if component.supportsService("com.sun.star.presentation.PresentationDocument"):
        return "Impress"
    return None

try:
    # Get LibreOffice context
    localContext = uno.getComponentContext()
    resolver = localContext.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver", localContext
    )

    # Connect to LibreOffice
    context = resolver.resolve(
        "uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext"
    )
    desktop = context.ServiceManager.createInstanceWithContext(
        "com.sun.star.frame.Desktop", context
    )

    # Execute the UNO code
{indented_code}

    print("UNO command executed successfully")

except Exception as e:
    print(f"UNO command failed: {{e}}")
    import traceback
    traceback.print_exc()
"""

    def _format_python_code(self, code: str) -> str:
        """
        Simple, robust Python code formatter for UNO/LLM-generated code.
        Handles common indentation issues without complex AST parsing.
        """
        import re

        # Remove markdown code blocks
        code = re.sub(r"```(?:python)?\s*", "", code)
        code = re.sub(r"```\s*$", "", code)

        # Split into lines, keeping empty lines for structure
        lines = code.split("\n")

        # Apply simple indentation
        formatted_lines = []
        indent_level = 0

        for line in lines:
            stripped = line.strip()

            # Skip empty lines and comments
            if not stripped or stripped.startswith("#"):
                continue

            # Dedent for elif/else/except/finally
            if stripped.startswith(("elif ", "else:", "except ", "finally:")):
                indent_level = max(0, indent_level - 1)

            # Add the line with current indentation
            formatted_lines.append("    " * indent_level + stripped)

            # Indent after structures ending with ':'
            if stripped.endswith(":"):
                indent_level += 1

        formatted_code = "\n".join(formatted_lines)

        # Validate syntax
        if self._validate_python_syntax(formatted_code):
            return formatted_code

        # Fallback: return cleaned but not formatted
        self.logger.warning("Formatted code has syntax errors, using unformatted version")
        return "\n".join(line.strip() for line in lines if line.strip() and not line.strip().startswith("#"))

    def _validate_python_syntax(self, code: str) -> bool:
        """Validate Python syntax by attempting to compile the code."""
        try:
            compile(code, "<string>", "exec")
            return True
        except SyntaxError:
            return False

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
            # Clean up partial connection on failure
            if self._playwright:
                try:
                    self._playwright.stop()
                except Exception as e:
                    pass
                self._playwright = None
                self._browser = None
                self._context = None
                self._page = None
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

        Returns:
            True if setup successful or already enabled, False on failure
        """
        try:
            self.logger.info("Setting up AT-SPI accessibility...")

            # Quick Python setup - minimal wait time
            python_code = """
import subprocess
import time

try:
    # Enable accessibility via gsettings
    subprocess.run(['gsettings', 'set', 'org.gnome.desktop.interface', 'toolkit-accessibility', 'true'],
                   capture_output=True, text=True, timeout=2)
    subprocess.run(['gsettings', 'set', 'org.gnome.desktop.interface', 'accessibility', 'true'],
                   capture_output=True, text=True, timeout=2)

    # Check if AT-SPI bus is running
    result = subprocess.run(['pgrep', '-x', 'at-spi-bus-launcher'],
                           capture_output=True, text=True, timeout=1)

    if result.returncode != 0:
        # Launch in background
        subprocess.Popen(['/usr/libexec/at-spi-bus-launcher', '--launch-immediately'],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print('AT-SPI bus launched')
    else:
        print('AT-SPI bus already active')

except Exception as e:
    print(f'AT-SPI setup warning: {e}')
"""

            result = self.execute_python_command(python_code)

            if result and result.get("status") == "success":
                output = result.get("output", "").strip()
                self.logger.info(f"AT-SPI setup completed: {output}")
                return True
            else:
                self.logger.warning(f"AT-SPI setup failed, accessibility may not work: {result}")
                return False

        except Exception as e:
            self.logger.error(f"Error setting up accessibility: {e}")
            return False

    def get_app_states(self, use_autoglm_enhancement: bool = True) -> list:
        """Get clean app states using autoglm_v tools"""
        if use_autoglm_enhancement:
            try:
                # Get accessibility tree for autoglm_v processing
                accessibility_tree = self.get_accessibility_tree()
                if accessibility_tree:
                    app_states = self._autoglm_extractor.extract_app_states(accessibility_tree)
                    if app_states:
                        self.logger.info(f"Extracted {len(app_states)} app states using autoglm_v")
                        return app_states

                # Fallback to clean extractor
                self.logger.info("Falling back to clean app state extractor")
                return self._app_state_extractor.extract_app_states(False)

            except Exception as e:
                self.logger.error(f"Error with autoglm_v app state extraction: {e}")
                return self._app_state_extractor.extract_app_states(False)
        else:
            return self._app_state_extractor.extract_app_states(False)

    def visualize_element_bounding_boxes(
        self, app_states: list, target_element_id: str = None, output_path: str = None
    ) -> str:
        """
        Visualize bounding boxes of extracted elements on screenshot for debugging.

        Args:
            app_states: List of AppState objects
            target_element_id: Specific element ID to highlight (optional)
            output_path: Path to save the annotated screenshot (optional)

        Returns:
            Path to the annotated screenshot
        """
        try:
            import os
            import time
            from io import BytesIO

            from PIL import Image, ImageDraw, ImageFont

            # Get current screenshot
            screenshot_data = self.get_screenshot()
            if not screenshot_data:
                self.logger.error("Could not get screenshot for visualization")
                return None

            # Convert screenshot to PIL Image
            screenshot = Image.open(BytesIO(screenshot_data))
            draw = ImageDraw.Draw(screenshot)

            # Try to load a font, fallback to default if not available
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
                small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
            except Exception:
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()

            colors = [
                (255, 0, 0),  # Red
                (0, 255, 0),  # Green
                (0, 0, 255),  # Blue
                (255, 255, 0),  # Yellow
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Cyan
            ]

            element_count = 0
            highlighted_element = None

            # Get current mouse position
            mouse_x, mouse_y = self._get_mouse_position()

            # Draw bounding boxes for all elements
            for app_state in app_states:
                if not hasattr(app_state, "elements"):
                    continue

                color = colors[element_count % len(colors)]

                for element in app_state.elements:
                    if not hasattr(element, "position"):
                        continue

                    pos = element.position
                    center_x = pos.get("center_x", 0)
                    center_y = pos.get("center_y", 0)
                    width = pos.get("width", 0)
                    height = pos.get("height", 0)

                    # Calculate bounding box coordinates
                    left = center_x - width // 2
                    top = center_y - height // 2
                    right = center_x + width // 2
                    bottom = center_y + height // 2

                    # Check if this is the target element
                    is_target = (
                        target_element_id
                        and hasattr(element, "element_id")
                        and element.element_id == target_element_id
                    )

                    if is_target:
                        highlighted_element = element
                        # Use bright red for target element
                        box_color = (255, 0, 0)
                        text_color = (255, 255, 255)
                        thickness = 3
                    else:
                        box_color = color
                        text_color = (255, 255, 255)
                        thickness = 1

                    # Draw bounding box
                    draw.rectangle([left, top, right, bottom], outline=box_color, width=thickness)

                    # Draw element label with coordinates
                    label = f"{element.name or element.element_type}"
                    if hasattr(element, "element_id"):
                        label = f"{str(element.element_id)[:4]}: {label}"

                    # Add coordinates to label
                    coord_label = f"({center_x}, {center_y})"

                    # Draw text background for main label
                    text_bbox = draw.textbbox((left, top - 35), label, font=font)
                    draw.rectangle(text_bbox, fill=box_color)

                    # Draw main label
                    draw.text((left, top - 35), label, fill=text_color, font=font)

                    # Draw coordinates label
                    coord_bbox = draw.textbbox((left, top - 20), coord_label, font=small_font)
                    draw.rectangle(coord_bbox, fill=(0, 0, 0, 180))
                    draw.text((left, top - 20), coord_label, fill=(255, 255, 255), font=small_font)

                    element_count += 1

            # Draw mouse position
            if mouse_x is not None and mouse_y is not None:
                # Draw mouse cursor as a cross
                cross_size = 15
                draw.line(
                    [mouse_x - cross_size, mouse_y, mouse_x + cross_size, mouse_y],
                    fill=(255, 255, 0),
                    width=3,
                )  # Yellow horizontal line
                draw.line(
                    [mouse_x, mouse_y - cross_size, mouse_x, mouse_y + cross_size],
                    fill=(255, 255, 0),
                    width=3,
                )  # Yellow vertical line

                # Draw mouse coordinates
                mouse_label = f"Mouse: ({mouse_x}, {mouse_y})"
                mouse_bbox = draw.textbbox((mouse_x + 20, mouse_y - 10), mouse_label, font=font)
                draw.rectangle(mouse_bbox, fill=(255, 255, 0, 180))  # Yellow background
                draw.text((mouse_x + 20, mouse_y - 10), mouse_label, fill=(0, 0, 0), font=font)

            # Add summary information
            summary_text = f"Total elements: {element_count}"
            if highlighted_element:
                summary_text += f"\nTarget: {highlighted_element.name} ({highlighted_element.element_id})"
                pos = highlighted_element.position
                summary_text += (
                    f"\nCoords: ({pos['center_x']}, {pos['center_y']}) size {pos['width']}x{pos['height']}"
                )

            if mouse_x is not None and mouse_y is not None:
                summary_text += f"\nMouse: ({mouse_x}, {mouse_y})"

            # Draw summary box
            draw.rectangle([10, 10, 350, 100], fill=(0, 0, 0, 180))
            draw.text((15, 15), summary_text, fill=(255, 255, 255), font=font)

            # Save annotated screenshot
            if not output_path:
                import tempfile

                output_path = os.path.join(
                    tempfile.gettempdir(), f"element_visualization_{int(time.time())}.png"
                )

            screenshot.save(output_path)
            self.logger.info(f"Element visualization saved to: {output_path}")

            return output_path

        except Exception as e:
            self.logger.error(f"Error creating element visualization: {e}")
            return None

    def _get_mouse_position(self) -> tuple:
        """Get current mouse position from the VM"""
        try:
            # Use the VM server's cursor position endpoint
            import requests

            response = requests.get(f"{self.http_server}/cursor_position", timeout=5)
            if response.status_code == 200:
                coords = response.json()
                if isinstance(coords, list) and len(coords) >= 2:
                    return (coords[0], coords[1])
            return (None, None)
        except Exception as e:
            self.logger.debug(f"Could not get mouse position: {e}")
            return (None, None)
