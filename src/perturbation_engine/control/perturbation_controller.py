"""
PerturbationController: Execute perturbation code
Clean interface for VM manipulation
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from playwright.sync_api import Page, sync_playwright

from OSWorld.desktop_env.controllers.python import PythonController
from OSWorld.desktop_env.controllers.setup import SetupController
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

    def __init__(
        self, vm_ip: str, server_port: int, chromium_port: int = 9222, client_password: str = "", **kwargs
    ):
        # Ensure logging is configured for subprocess (only if not already configured)
        if not logging.getLogger().handlers:
            from perturbation_engine.configure_logging import configure_logging

            configure_logging()

        PythonController.__init__(self, vm_ip, server_port, **kwargs)
        SetupController.__init__(self, vm_ip, server_port, chromium_port, **kwargs)
        self.vm_ip = vm_ip
        self.server_port = server_port
        self.chromium_port = chromium_port
        self.client_password = client_password
        self.logger = logging.getLogger(__name__)

        # Playwright connection
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

        self._autoglm_extractor = AutoglmAppStateExtractor(controller=self)

        # Track launched apps with enhanced capabilities
        self._launched_apps = {"chrome": False, "vscode": False, "libreoffice": False}

        # Setup X11 tools for enhanced window management
        self._setup_x11_tools()

    def _setup_x11_tools(self) -> bool:
        """
        Setup X11 tools required for enhanced window management and app state extraction.

        Uses deterministic installation with sudo -S pattern for consistent server setup.

        Returns:
            True if setup successful, False on failure
        """
        try:
            self.logger.info("Setting up X11 tools for enhanced window management...")

            # Define required X11 packages for deterministic setup
            x11_packages = [
                "x11-utils",  # Contains xprop, xwininfo, xdpyinfo
                "xdotool",  # Window manipulation tool
                "wmctrl",  # Window manager control
                "xclip",  # Clipboard utilities
                "socat",  # Network utility for port forwarding
                "gnome-screenshot",  # Screenshot utility
                "ffmpeg",  # Video recording
                "python3-tk",  # Python Tkinter support
                "python3-dev",  # Python development headers
            ]

            # Use deterministic installation pattern like setup.py
            install_command = f"""
import subprocess
import sys

def install_x11_packages():
    packages = {x11_packages}
    client_password = "{getattr(self, "client_password", "")}"

    # Update package list first
    update_cmd = f"echo '{{client_password}}' | sudo -S bash -c \\"apt-get update\\""
    result = subprocess.run(update_cmd, shell=True, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        print(f"Failed to update package list: {{result.stderr}}")
        return False

    # Install each package deterministically
    failed_packages = []
    for package in packages:
        try:
            # Check if package is already installed
            check_cmd = f"echo '{{client_password}}' | sudo -S bash -c \\"dpkg -l {{package}}\\""
            check_result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True, timeout=10)

            if check_result.returncode == 0 and 'ii' in check_result.stdout:
                print(f'Package {{package}} already installed')
                continue

            # Install package deterministically
            print(f'Installing {{package}}...')
            install_cmd = f"echo '{{client_password}}' | sudo -S bash -c \\"apt-get install -y {{package}}\\""
            result = subprocess.run(install_cmd, shell=True, capture_output=True, text=True, timeout=120)

            if result.returncode == 0:
                print(f'Successfully installed {{package}}')
            else:
                print(f'Failed to install {{package}}: {{result.stderr}}')
                failed_packages.append(package)

        except Exception as e:
            print(f'Error installing {{package}}: {{e}}')
            failed_packages.append(package)

    if failed_packages:
        print(f'Failed to install packages: {{failed_packages}}')
        return False
    else:
        print('All X11 packages installed successfully')
        return True

install_x11_packages()
"""

            result = self.execute_python_command(install_command)

            if result and result.get("status") == "success":
                output = result.get("output", "").strip()
                self.logger.info(f"X11 tools installation completed: {output}")

                # Verify installation deterministically
                verification_command = """
import subprocess

def verify_x11_tools():
    tools = ['xprop', 'xdotool', 'xwininfo', 'wmctrl']
    available_tools = []

    for tool in tools:
        try:
            result = subprocess.run([tool, '--version'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                available_tools.append(tool)
                print(f'{tool}: available')
            else:
                print(f'{tool}: not available')
        except Exception as e:
            print(f'{tool}: error - {e}')

    print(f'Available tools: {available_tools}')
    return len(available_tools) >= 3  # Require at least 3 tools for deterministic setup

verify_x11_tools()
"""

                verify_result = self.execute_python_command(verification_command)
                if verify_result and verify_result.get("status") == "success":
                    output = verify_result.get("output", "").strip()
                    self.logger.info(f"X11 tools verification: {output}")

                    # Check if we have the minimum required tools
                    if (
                        "xprop" in output
                        and "xwininfo" in output
                        and ("xdotool" in output or "wmctrl" in output)
                    ):
                        self.logger.info("X11 tools setup completed successfully")
                        return True
                    else:
                        self.logger.error("X11 tools verification failed - insufficient tools available")
                        return False
                else:
                    self.logger.error("X11 tools verification failed")
                    return False
            else:
                self.logger.error(f"X11 tools installation failed: {result}")
                return False

        except Exception as e:
            self.logger.error(f"Error setting up X11 tools: {e}")
            return False

    def execute_perturbation(
        self, perturbation_type: str, generated_code: str, api_call: str, parameters: Dict[str, Any]
    ) -> ManipulationResult:
        """Execute perturbation using generated code with comprehensive operation support"""
        try:
            success = False
            result_data = {}

            # Core execution methods
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

            # Visual manipulation operations
            elif api_call == "execute_css_injection":
                success = self.execute_css_injection(generated_code, parameters)
                result_data = {"api_call": api_call, "css": generated_code, "parameters": parameters}
            elif api_call == "execute_dom_modification":
                success = self.execute_dom_modification(generated_code, parameters)
                result_data = {"api_call": api_call, "dom_code": generated_code, "parameters": parameters}
            elif api_call == "execute_theme_randomization":
                success = self.execute_theme_randomization(parameters)
                result_data = {"api_call": api_call, "parameters": parameters}
            elif api_call == "execute_layout_perturbation":
                success = self.execute_layout_perturbation(parameters)
                result_data = {"api_call": api_call, "parameters": parameters}
            elif api_call == "execute_typography_randomization":
                success = self.execute_typography_randomization(parameters)
                result_data = {"api_call": api_call, "parameters": parameters}
            elif api_call == "execute_animation_effects":
                success = self.execute_animation_effects(generated_code, parameters)
                result_data = {
                    "api_call": api_call,
                    "animation_code": generated_code,
                    "parameters": parameters,
                }
            elif api_call == "execute_accessibility_perturbation":
                success = self.execute_accessibility_perturbation(parameters)
                result_data = {"api_call": api_call, "parameters": parameters}

            # Freeform operations
            elif api_call == "execute_python_execution":
                success = self.execute_python_execution(generated_code, parameters)
                result_data = {"api_call": api_call, "python_code": generated_code, "parameters": parameters}
            elif api_call == "execute_javascript_injection":
                success = self.execute_javascript_injection(generated_code, parameters)
                result_data = {"api_call": api_call, "js_code": generated_code, "parameters": parameters}
            elif api_call == "execute_bash_automation":
                success = self.execute_bash_automation(generated_code, parameters)
                result_data = {"api_call": api_call, "bash_code": generated_code, "parameters": parameters}
            elif api_call == "execute_playwright_automation":
                success = self.execute_playwright_automation(generated_code, parameters)
                result_data = {
                    "api_call": api_call,
                    "playwright_code": generated_code,
                    "parameters": parameters,
                }
            elif api_call == "execute_file_system_manipulation":
                success = self.execute_file_system_manipulation(parameters)
                result_data = {"api_call": api_call, "parameters": parameters}
            elif api_call == "execute_network_perturbation":
                success = self.execute_network_perturbation(parameters)
                result_data = {"api_call": api_call, "parameters": parameters}
            elif api_call == "execute_system_integration":
                success = self.execute_system_integration(parameters)
                result_data = {"api_call": api_call, "parameters": parameters}

            # Legacy operations
            elif api_call == "manipulate_app_state":
                success = self._manipulate_app_state(parameters)
                result_data = {"api_call": api_call, "parameters": parameters}
            elif api_call == "execute_system_perturbation":
                system_type = parameters.get("system_type", "desktop_theme")
                success = self.execute_system_perturbation(system_type, parameters)
                result_data = {"api_call": api_call, "system_type": system_type, "parameters": parameters}
            else:
                raise ValueError(f"Unsupported API call: {api_call}")

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
        # Clean and format the UNO code properly
        lines = formatted_uno_code.strip().split("\n")
        indented_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped:  # Only process non-empty lines
                # Add proper indentation for the try block
                indented_lines.append("    " + stripped)

        indented_code = "\n".join(indented_lines)

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
                raise ValueError(f"Unknown system perturbation type: {perturbation_type}")

        except Exception as e:
            self.logger.error(f"Error executing system perturbation: {e}")
            return False

    def execute_css_injection(self, css_code: str, parameters: Dict[str, Any]) -> bool:
        """Execute CSS injection for visual manipulation"""
        try:
            page = self._get_page()
            if not page:
                return False

            target_selector = parameters.get("target_selector", "body")
            js_code = f"""
            const style = document.createElement('style');
            style.textContent = `{css_code}`;
            document.querySelector('{target_selector}').appendChild(style);
            """

            page.evaluate(js_code)
            return True
        except Exception as e:
            self.logger.error(f"Error executing CSS injection: {e}")
            return False

    def execute_dom_modification(self, dom_code: str, parameters: Dict[str, Any]) -> bool:
        """Execute DOM modification for element manipulation"""
        try:
            page = self._get_page()
            if not page:
                return False

            page.evaluate(dom_code)
            return True
        except Exception as e:
            self.logger.error(f"Error executing DOM modification: {e}")
            return False

    def execute_theme_randomization(self, parameters: Dict[str, Any]) -> bool:
        """Execute theme randomization"""
        try:
            _color_palette = parameters.get("color_palette", "random")
            theme_variant = parameters.get("theme_variant", "dark")
            accent_colors = parameters.get("accent_colors", ["#ff6b6b", "#4ecdc4", "#45b7d1"])

            css_code = f"""
            :root {{
                --primary-color: {accent_colors[0]};
                --secondary-color: {accent_colors[1]};
                --accent-color: {accent_colors[2]};
                --background-color: {"#1a1a1a" if theme_variant == "dark" else "#ffffff"};
                --text-color: {"#ffffff" if theme_variant == "dark" else "#000000"};
            }}
            """

            return self.execute_css_injection(css_code, parameters)
        except Exception as e:
            self.logger.error(f"Error executing theme randomization: {e}")
            return False

    def execute_layout_perturbation(self, parameters: Dict[str, Any]) -> bool:
        """Execute layout perturbation"""
        try:
            element_selector = parameters.get("element_selector", ".main-content")
            position_changes = parameters.get("position_changes", {})

            css_code = f"""
            {element_selector} {{
                transform: translate({position_changes.get("x", 0)}px, {position_changes.get("y", 0)}px);
                width: {position_changes.get("width", "auto")};
                height: {position_changes.get("height", "auto")};
            }}
            """

            return self.execute_css_injection(css_code, parameters)
        except Exception as e:
            self.logger.error(f"Error executing layout perturbation: {e}")
            return False

    def execute_typography_randomization(self, parameters: Dict[str, Any]) -> bool:
        """Execute typography randomization"""
        try:
            font_family = parameters.get("font_family", "Arial, sans-serif")
            font_size = parameters.get("font_size", "14px")
            font_weight = parameters.get("font_weight", "normal")

            css_code = f"""
            body {{
                font-family: {font_family};
                font-size: {font_size};
                font-weight: {font_weight};
            }}
            """

            return self.execute_css_injection(css_code, parameters)
        except Exception as e:
            self.logger.error(f"Error executing typography randomization: {e}")
            return False

    def execute_animation_effects(self, animation_code: str, parameters: Dict[str, Any]) -> bool:
        """Execute animation effects"""
        try:
            page = self._get_page()
            if not page:
                return False

            page.evaluate(animation_code)
            return True
        except Exception as e:
            self.logger.error(f"Error executing animation effects: {e}")
            return False

    def execute_accessibility_perturbation(self, parameters: Dict[str, Any]) -> bool:
        """Execute accessibility perturbation"""
        try:
            aria_labels = parameters.get("aria_labels", {})
            contrast_ratio = parameters.get("contrast_ratio", "normal")

            js_code = f"""
            Object.entries({aria_labels}).forEach(([selector, label]) => {{
                const element = document.querySelector(selector);
                if (element) {{
                    element.setAttribute('aria-label', label);
                }}
            }});
            """

            page = self._get_page()
            if page:
                page.evaluate(js_code)

            if contrast_ratio != "normal":
                css_code = f"body {{ filter: contrast({contrast_ratio}); }}"
                self.execute_css_injection(css_code, parameters)

            return True
        except Exception as e:
            self.logger.error(f"Error executing accessibility perturbation: {e}")
            return False

    def execute_python_execution(self, python_code: str, parameters: Dict[str, Any]) -> bool:
        """Execute Python code with enhanced capabilities"""
        try:
            result = self.execute_python_command(python_code)
            return result.get("status") == "success"
        except Exception as e:
            self.logger.error(f"Error executing Python code: {e}")
            return False

    def execute_javascript_injection(self, js_code: str, parameters: Dict[str, Any]) -> bool:
        """Execute JavaScript injection with enhanced capabilities"""
        try:
            page = self._get_page()
            if not page:
                return False

            page.evaluate(js_code)
            return True
        except Exception as e:
            self.logger.error(f"Error executing JavaScript injection: {e}")
            return False

    def execute_bash_automation(self, bash_code: str, parameters: Dict[str, Any]) -> bool:
        """Execute bash automation with enhanced capabilities"""
        try:
            return self.execute_bash_command(bash_code)
        except Exception as e:
            self.logger.error(f"Error executing bash automation: {e}")
            return False

    def execute_playwright_automation(self, playwright_code: str, parameters: Dict[str, Any]) -> bool:
        """Execute Playwright automation"""
        try:
            page = self._get_page()
            if not page:
                return False

            page.evaluate(playwright_code)
            return True
        except Exception as e:
            self.logger.error(f"Error executing Playwright automation: {e}")
            return False

    def execute_file_system_manipulation(self, parameters: Dict[str, Any]) -> bool:
        """Execute file system manipulation"""
        try:
            operation = parameters.get("operation", "create_file")
            file_path = parameters.get("file_path", "/tmp/perturbation_file")
            content = parameters.get("content", "Perturbation content")

            if operation == "create_file":
                python_code = f"""
                with open('{file_path}', 'w') as f:
                    f.write('{content}')
                """
                return self.execute_python_command(python_code).get("status") == "success"
            elif operation == "modify_file":
                python_code = f"""
                with open('{file_path}', 'a') as f:
                    f.write('\\n{content}')
                """
                return self.execute_python_command(python_code).get("status") == "success"
            else:
                raise ValueError(f"Unknown file system operation: {operation}")
        except Exception as e:
            self.logger.error(f"Error executing file system manipulation: {e}")
            return False

    def execute_network_perturbation(self, parameters: Dict[str, Any]) -> bool:
        """Execute network perturbation"""
        try:
            perturbation_type = parameters.get("perturbation_type", "delay")
            delay_ms = parameters.get("delay_ms", 1000)

            if perturbation_type == "delay":
                js_code = f"""
                const originalFetch = window.fetch;
                window.fetch = function(...args) {{
                    return new Promise(resolve => {{
                        setTimeout(() => {{
                            resolve(originalFetch.apply(this, args));
                        }}, {delay_ms});
                    }});
                }};
                """
                return self.execute_javascript_injection(js_code, parameters)
            else:
                raise ValueError(f"Unknown network perturbation type: {perturbation_type}")
        except Exception as e:
            self.logger.error(f"Error executing network perturbation: {e}")
            return False

    def execute_system_integration(self, parameters: Dict[str, Any]) -> bool:
        """Execute system integration operations"""
        try:
            operation = parameters.get("operation", "modify_settings")
            setting_key = parameters.get("setting_key", "org.gnome.desktop.interface.gtk-theme")
            setting_value = parameters.get("setting_value", "Adwaita-dark")

            if operation == "modify_settings":
                command = f"gsettings set {setting_key} '{setting_value}'"
                return self.execute_bash_command(command)
            else:
                raise ValueError(f"Unknown system integration operation: {operation}")
        except Exception as e:
            self.logger.error(f"Error executing system integration: {e}")
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
        try:
            if use_autoglm_enhancement:
                return self.get_enhanced_app_states()
            else:
                # Fallback to basic extraction
                accessibility_tree = self.get_accessibility_tree()
                if accessibility_tree:
                    app_states = self._autoglm_extractor.extract_app_states(accessibility_tree)
                    if app_states:
                        self.logger.info(f"Extracted {len(app_states)} app states")
                        return app_states

        except Exception as e:
            self.logger.exception(f"Error with app state extraction: {e}")
            return []

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

    def launch_chrome_with_cdp(self, url: str = None) -> bool:
        """Launch Chrome with CDP debugging enabled"""
        try:
            if self._launched_apps["chrome"]:
                self.logger.info("Chrome already launched with CDP")
                return True

            # Build Chrome command with CDP flags
            chrome_cmd = [
                "google-chrome",
                f"--remote-debugging-port={self.chromium_port}",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-web-security",
                "--disable-features=VizDisplayCompositor",
                "--user-data-dir=/tmp/chrome-debug",
            ]

            if url:
                chrome_cmd.append(url)

            # Launch Chrome
            result = self.execute_python_command(f"""
import subprocess
import time
import requests

try:
    # Kill any existing Chrome processes
    subprocess.run(['pkill', '-f', 'google-chrome'], capture_output=True)
    time.sleep(1)

    # Launch Chrome with CDP
    subprocess.Popen({chrome_cmd}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Wait for Chrome to start and CDP to be available
    for i in range(10):
        try:
            response = requests.get(f'http://localhost:{self.chromium_port}/json', timeout=1)
            if response.status_code == 200:
                print('Chrome launched with CDP successfully')
                break
        except:
            time.sleep(1)
    else:
        print('Chrome CDP setup failed')
""")

            if result.get("status") == "success":
                self._launched_apps["chrome"] = True
                self.logger.info("Chrome launched with CDP debugging enabled")
                return True
            else:
                self.logger.warning("Failed to launch Chrome with CDP")
                return False

        except Exception as e:
            self.logger.error(f"Error launching Chrome with CDP: {e}")
            return False

    def launch_vscode_with_cdp(self, path: str = None) -> bool:
        """Launch VS Code with CDP debugging enabled"""
        try:
            if self._launched_apps["vscode"]:
                self.logger.info("VS Code already launched with CDP")
                return True

            # Build VS Code command with CDP flags
            vscode_cmd = ["code", "--inspect-extensions=9229"]
            if path:
                vscode_cmd.append(path)

            # Launch VS Code
            result = self.execute_python_command(f"""
import subprocess
import time
import requests

try:
    # Kill any existing VS Code processes
    subprocess.run(['pkill', '-f', 'code'], capture_output=True)
    time.sleep(1)

    # Launch VS Code with CDP
    subprocess.Popen({vscode_cmd}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Wait for VS Code to start
    time.sleep(3)
    print('VS Code launched with CDP debugging')
""")

            if result.get("status") == "success":
                self._launched_apps["vscode"] = True
                self.logger.info("VS Code launched with CDP debugging enabled")
                return True
            else:
                self.logger.warning("Failed to launch VS Code with CDP")
                return False

        except Exception as e:
            self.logger.error(f"Error launching VS Code with CDP: {e}")
            return False

    def launch_libreoffice_with_uno(self, app_type: str = "calc", file_path: str = None) -> bool:
        """Launch LibreOffice with UNO API enabled"""
        try:
            if self._launched_apps["libreoffice"]:
                self.logger.info("LibreOffice already launched with UNO")
                return True

            # Build LibreOffice command with UNO flags
            if app_type == "calc":
                app_cmd = ["libreoffice", "--calc"]
            elif app_type == "writer":
                app_cmd = ["libreoffice", "--writer"]
            elif app_type == "impress":
                app_cmd = ["libreoffice", "--impress"]
            else:
                app_cmd = ["libreoffice"]

            # Add UNO socket flag
            app_cmd.extend(
                ["--accept=socket,host=localhost,port=2002;urp;StarOffice.ServiceManager", "--headless"]
            )

            if file_path:
                app_cmd.append(file_path)

            # Launch LibreOffice
            result = self.execute_python_command(f"""
import subprocess
import time
import socket

try:
    # Kill any existing LibreOffice processes
    subprocess.run(['pkill', '-f', 'libreoffice'], capture_output=True)
    time.sleep(1)

    # Launch LibreOffice with UNO
    subprocess.Popen({app_cmd}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Wait for UNO socket to be available
    for i in range(10):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', 2002))
            sock.close()
            if result == 0:
                print('LibreOffice launched with UNO successfully')
                break
        except:
            time.sleep(1)
    else:
        print('LibreOffice UNO setup failed')
""")

            if result.get("status") == "success":
                self._launched_apps["libreoffice"] = True
                self.logger.info(f"LibreOffice {app_type} launched with UNO API enabled")
                return True
            else:
                self.logger.warning("Failed to launch LibreOffice with UNO")
                return False

        except Exception as e:
            self.logger.error(f"Error launching LibreOffice with UNO: {e}")
            return False

    def ensure_app_enhancement_setup(self) -> Dict[str, bool]:
        """Ensure all target apps are launched with enhancement capabilities"""
        setup_results = {}

        try:
            # Setup Chrome with CDP
            setup_results["chrome"] = self.launch_chrome_with_cdp()

            # Setup VS Code with CDP
            setup_results["vscode"] = self.launch_vscode_with_cdp()

            # Setup LibreOffice with UNO
            setup_results["libreoffice"] = self.launch_libreoffice_with_uno()

            self.logger.info(f"App enhancement setup results: {setup_results}")
            return setup_results

        except Exception as e:
            self.logger.error(f"Error in app enhancement setup: {e}")
            return {"chrome": False, "vscode": False, "libreoffice": False}

    def integrate_existing_tools(self) -> Dict[str, Any]:
        """Integrate with existing autoglm_v package tools where useful"""
        try:
            from perturbation_engine.tools.autoglm_v.tools.package.code import CodeTools
            from perturbation_engine.tools.autoglm_v.tools.package.google_chrome import BrowserTools
            from perturbation_engine.tools.autoglm_v.tools.package.vlc import VLCTools

            # Store references to existing tools
            self._existing_tools = {"chrome": BrowserTools, "code": CodeTools, "vlc": VLCTools}

            self.logger.info("Integrated existing autoglm_v package tools")
            return {"status": "success", "tools": list(self._existing_tools.keys())}

        except Exception as e:
            self.logger.warning(f"Could not integrate existing tools: {e}")
            return {"status": "error", "error": str(e)}

    def get_enhanced_app_states(self) -> List[Dict[str, Any]]:
        """Get enhanced app states using the improved extractor"""
        try:
            # Get accessibility tree
            accessibility_tree = self.get_accessibility_tree()
            if not accessibility_tree:
                self.logger.warning("No accessibility tree available")
                return []

            # Extract app states with X11+CDP+UNO enhancement
            app_states = self._autoglm_extractor.extract_app_states(accessibility_tree)

            if app_states:
                self.logger.info(f"Extracted {len(app_states)} enhanced app states")
                return app_states
            else:
                self.logger.warning("No app states extracted")
                return []

        except Exception as e:
            self.logger.error(f"Error getting enhanced app states: {e}")
            return []

    def get_chrome_dom_data(self) -> Dict[str, Any]:
        """Get Chrome DOM data using CDP"""
        try:
            page = self._get_page()
            if not page:
                self.logger.warning("No Chrome page available for DOM extraction")
                return {}

            # Extract comprehensive DOM data
            dom_data = page.evaluate("""
                () => {
                    const data = {
                        url: window.location.href,
                        title: document.title,
                        buttons: [],
                        links: [],
                        inputs: [],
                        forms: [],
                        tables: [],
                        images: [],
                        meta: {
                            viewport: document.querySelector('meta[name="viewport"]')?.content || '',
                            description: document.querySelector('meta[name="description"]')?.content || ''
                        }
                    };

                    // Extract all interactive elements
                    document.querySelectorAll('button, input[type="button"], input[type="submit"], input[type="reset"]').forEach((btn, i) => {
                        const rect = btn.getBoundingClientRect();
                        data.buttons.push({
                            id: btn.id || `button_${i}`,
                            text: btn.textContent?.trim() || btn.value || '',
                            class: btn.className,
                            type: btn.type || 'button',
                            aria_label: btn.getAttribute('aria-label'),
                            disabled: btn.disabled,
                            visible: rect.width > 0 && rect.height > 0,
                            position: {
                                x: Math.round(rect.left),
                                y: Math.round(rect.top),
                                width: Math.round(rect.width),
                                height: Math.round(rect.height),
                                center_x: Math.round(rect.left + rect.width / 2),
                                center_y: Math.round(rect.top + rect.height / 2)
                            }
                        });
                    });

                    // Extract links
                    document.querySelectorAll('a[href]').forEach((link, i) => {
                        const rect = link.getBoundingClientRect();
                        data.links.push({
                            id: link.id || `link_${i}`,
                            text: link.textContent?.trim() || '',
                            href: link.href,
                            target: link.target || '_self',
                            visible: rect.width > 0 && rect.height > 0,
                            position: {
                                x: Math.round(rect.left),
                                y: Math.round(rect.top),
                                width: Math.round(rect.width),
                                height: Math.round(rect.height),
                                center_x: Math.round(rect.left + rect.width / 2),
                                center_y: Math.round(rect.top + rect.height / 2)
                            }
                        });
                    });

                    // Extract input fields
                    document.querySelectorAll('input, textarea, select').forEach((input, i) => {
                        const rect = input.getBoundingClientRect();
                        data.inputs.push({
                            id: input.id || `input_${i}`,
                            type: input.type || input.tagName.toLowerCase(),
                            name: input.name || '',
                            placeholder: input.placeholder || '',
                            value: input.value || '',
                            required: input.required || false,
                            disabled: input.disabled || false,
                            visible: rect.width > 0 && rect.height > 0,
                            position: {
                                x: Math.round(rect.left),
                                y: Math.round(rect.top),
                                width: Math.round(rect.width),
                                height: Math.round(rect.height),
                                center_x: Math.round(rect.left + rect.width / 2),
                                center_y: Math.round(rect.top + rect.height / 2)
                            }
                        });
                    });

                    return data;
                }
            """)

            self.logger.info(
                f"Extracted Chrome DOM data: {len(dom_data.get('buttons', []))} buttons, {len(dom_data.get('links', []))} links, {len(dom_data.get('inputs', []))} inputs"
            )
            return dom_data

        except Exception as e:
            self.logger.error(f"Error extracting Chrome DOM data: {e}")
            return {}

    def get_libreoffice_state(self, app_type: str = "calc") -> Dict[str, Any]:
        """Get LibreOffice state using UNO API"""
        try:
            if app_type == "calc":
                uno_code = """
# Get comprehensive Calc document state
doc = desktop.getCurrentComponent()
if doc and doc.supportsService("com.sun.star.sheet.SpreadsheetDocument"):
    sheets = doc.getSheets()
    active_sheet = doc.getCurrentController().getActiveSheet()
    current_cell = doc.getCurrentController().getActiveCell()
    cell_address = current_cell.getCellAddress()

    # Get sheet information
    sheet_names = []
    for i in range(sheets.getCount()):
        sheet_names.append(sheets.getByIndex(i).getName())

    # Get current cell information
    cell_value = current_cell.getFormula()
    cell_type = current_cell.getType()

    # Get document properties
    doc_props = doc.getDocumentInfo()

    print(f"SHEETS: {sheet_names}")
    print(f"ACTIVE_SHEET: {active_sheet.getName()}")
    print(f"CURRENT_CELL: {cell_address.Column},{cell_address.Row}")
    print(f"CELL_VALUE: {cell_value}")
    print(f"CELL_TYPE: {cell_type}")
    print(f"DOCUMENT_TITLE: {doc.getTitle()}")
    print(f"DOCUMENT_URL: {doc.getURL()}")
    print(f"HAS_LOCATION: {doc.hasLocation()}")
    print(f"DOCUMENT_MODIFIED: {doc.isModified()}")
else:
    print("NO_CALC_DOCUMENT")
"""
            elif app_type == "writer":
                uno_code = """
# Get comprehensive Writer document state
doc = desktop.getCurrentComponent()
if doc and doc.supportsService("com.sun.star.text.TextDocument"):
    text = doc.Text
    cursor = text.createTextCursor()

    # Get document information
    print(f"DOCUMENT_TITLE: {doc.getTitle()}")
    print(f"DOCUMENT_URL: {doc.getURL()}")
    print(f"HAS_LOCATION: {doc.hasLocation()}")
    print(f"DOCUMENT_MODIFIED: {doc.isModified()}")

    # Get current selection/text
    if cursor.getString():
        print(f"CURRENT_TEXT: {cursor.getString()[:200]}")
        print(f"TEXT_LENGTH: {len(cursor.getString())}")
    else:
        print("NO_SELECTION")

    # Get page count
    print(f"PAGE_COUNT: {doc.getPageCount()}")
else:
    print("NO_WRITER_DOCUMENT")
"""
            else:
                uno_code = """
# Get generic LibreOffice document state
doc = desktop.getCurrentComponent()
if doc:
    print(f"DOCUMENT_TITLE: {doc.getTitle()}")
    print(f"DOCUMENT_URL: {doc.getURL()}")
    print(f"HAS_LOCATION: {doc.hasLocation()}")
    print(f"DOCUMENT_MODIFIED: {doc.isModified()}")
    print(f"DOCUMENT_TYPE: {doc.getClass().getName()}")
else:
    print("NO_DOCUMENT")
"""

            # Execute UNO code
            result = self.execute_uno_command(uno_code, {})

            if result and result.get("status") == "success":
                output = result.get("output", "")
                parsed_data = self._parse_libreoffice_output(output, app_type)
                self.logger.info(f"Extracted LibreOffice {app_type} state successfully")
                return parsed_data
            else:
                self.logger.warning(f"Failed to get LibreOffice {app_type} state: {result}")
                return {}

        except Exception as e:
            self.logger.error(f"Error getting LibreOffice {app_type} state: {e}")
            return {}

    def _parse_libreoffice_output(self, output: str, app_type: str) -> Dict[str, Any]:
        """Parse LibreOffice UNO output into structured data"""
        data = {}

        try:
            lines = output.strip().split("\n")
            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip()

                    if key == "SHEETS":
                        # Parse sheet names from list format
                        if value.startswith("[") and value.endswith("]"):
                            sheets_str = value[1:-1]  # Remove brackets
                            data["sheets"] = [s.strip().strip("'\"") for s in sheets_str.split(",")]
                        else:
                            data["sheets"] = [value]
                    elif key == "ACTIVE_SHEET":
                        data["active_sheet"] = value
                    elif key == "CURRENT_CELL":
                        if "," in value:
                            col, row = value.split(",")
                            data["current_cell"] = {"column": int(col), "row": int(row)}
                    elif key == "CELL_VALUE":
                        data["cell_value"] = value
                    elif key == "CELL_TYPE":
                        data["cell_type"] = value
                    elif key == "DOCUMENT_TITLE":
                        data["document_title"] = value
                    elif key == "DOCUMENT_URL":
                        data["document_url"] = value
                    elif key == "HAS_LOCATION":
                        data["has_location"] = value.lower() == "true"
                    elif key == "DOCUMENT_MODIFIED":
                        data["document_modified"] = value.lower() == "true"
                    elif key == "CURRENT_TEXT":
                        data["current_text"] = value
                    elif key == "TEXT_LENGTH":
                        data["text_length"] = int(value) if value.isdigit() else 0
                    elif key == "PAGE_COUNT":
                        data["page_count"] = int(value) if value.isdigit() else 0
                    elif key == "DOCUMENT_TYPE":
                        data["document_type"] = value
                    else:
                        data[key.lower()] = value

            return data

        except Exception as e:
            self.logger.error(f"Error parsing LibreOffice output: {e}")
            return {}
