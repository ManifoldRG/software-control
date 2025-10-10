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
from perturbation_engine.pipeline.data_models import WindowState
from perturbation_engine.tools.autoglm_integration import AutoglmAppStateExtractor

# Import existing autoglm_v logic
from perturbation_engine.tools.autoglm_v.tools.package.code import CodeTools
from perturbation_engine.tools.autoglm_v.tools.package.google_chrome import BrowserTools
from perturbation_engine.tools.autoglm_v.tools.package.vlc import VLCTools

# Import LibreOffice tools (with fallback if not available)
try:
    from perturbation_engine.tools.autoglm_v.tools.package.libreoffice_calc import CalcTools
    from perturbation_engine.tools.autoglm_v.tools.package.libreoffice_impress import ImpressTools
    from perturbation_engine.tools.autoglm_v.tools.package.libreoffice_writer import WriterTools
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

        self._extractor = AutoglmAppStateExtractor(controller=self)
        self._setup_accessibility()

    def _setup_accessibility(self) -> bool:
        self._setup_x11_tools()
        accessibility_ok = self.ensure_accessibility_enabled()
        if accessibility_ok:
            self.logger.info("AT-SPI accessibility enabled successfully")
        else:
            self.logger.warning("AT-SPI accessibility may not be fully enabled - will retry later if needed")

    def _setup_x11_tools(self) -> bool:
        """Setup X11 tools for window management."""
        # Check if already set up to avoid duplicate setup
        if hasattr(self, "_x11_tools_setup") and self._x11_tools_setup:
            self.logger.info("X11 tools already set up, skipping...")
            return True

        try:
            self.logger.info("Setting up X11 tools...")

            packages = ["x11-utils", "xdotool", "wmctrl", "xclip"]
            # pwd = getattr(self, 'client_password', '')
            pwd = "password"

            # Install packages
            install_cmd = f"""
import subprocess, os, platform
env = os.environ.copy()
env['DEBIAN_FRONTEND'] = 'noninteractive'
pwd = "{pwd}"

# Detect architecture and fix repos if needed
arch = platform.machine()
if arch in ['aarch64', 'arm64']:
    print(f"Detected ARM64 architecture: {{arch}}")
    # Fix repository configuration for ARM64
    fix_repos_cmd = ['sudo', '-S', 'sed', '-i',
                    's/http:\\/\\/.*\\.archive\\.ubuntu\\.com/http:\\/\\/ports.ubuntu.com/g',
                    '/etc/apt/sources.list']
    proc = subprocess.Popen(fix_repos_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, text=True, env=env)
    proc.communicate(input=f"{{pwd}}\\n", timeout=30)
    print("Fixed ARM64 repository configuration")
else:
    print(f"Detected x86_64 architecture: {{arch}}")

# Update and install
for cmd in [
    ['sudo', '-S', 'apt-get', 'update', '-y'],
    ['sudo', '-S', 'apt-get', 'install', '-y', '--fix-missing'] + {packages}
]:
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, text=True, env=env)
    out, err = proc.communicate(input=f"{{pwd}}\\n", timeout=180)
    print(out)
    if proc.returncode != 0:
        print(f"Error: {{err}}")
        # Only fail on install errors, not update warnings
        if 'install' in ' '.join(cmd):
            exit(1)
print("Installation complete")
"""

            result = self.execute_python_command(install_cmd)
            if not result or result.get("status") != "success":
                self.logger.error(f"Installation failed: {result}")
                return False

            # Verify tools with better error reporting
            verify_cmd = """
import shutil
tools = ['xprop', 'xdotool', 'xwininfo', 'wmctrl']
available = [t for t in tools if shutil.which(t)]
missing = [t for t in tools if t not in available]

print(f"Available X11 tools: {available}")
print(f"Missing X11 tools: {missing}")

if len(available) >= 2:
    print("SUCCESS: Enough X11 tools available for window management")
    exit(0)
else:
    print(f"WARNING: Only {len(available)} X11 tools available, need at least 2")
    print("X11 tools setup may be incomplete - some window operations may fail")
    exit(1)
"""

            verify_result = self.execute_python_command(verify_cmd)
            if verify_result and verify_result.get("status") == "success":
                output = verify_result.get("output", "")
                self.logger.info(f"X11 tools verification successful: {output}")
                self._x11_tools_setup = True  # Mark as set up
                return True

            # Log verification failure with details
            output = verify_result.get("output", "No output") if verify_result else "No result"
            self.logger.warning(f"X11 tools verification failed: {output}")
            self.logger.warning("Some window management operations may not work properly")
            return False

        except Exception as e:
            self.logger.error(f"X11 tools setup error: {e}")
            self._x11_tools_setup = False  # Mark setup as failed
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

    def visualize_element_bounding_boxes(
        self, window_states: List[WindowState], target_element_id: str = None, output_path: str = None
    ) -> str:
        """
        Visualize bounding boxes of extracted elements on screenshot for debugging.

        Args:
            window_states: List of WindowState objects
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
            for window_state in window_states:
                elements = window_state.get_all_elements(include_structural=False)
                color = colors[element_count % len(colors)]

                for element in elements:
                    pos = element.position
                    if not pos:
                        continue

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
                        target_element_id and element.element_id and element.element_id == target_element_id
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
                    label = element.name or element.element_type or "Unknown"
                    if element.element_id:
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
                summary_text += (
                    f"\nTarget: {highlighted_element.name or 'Unknown'} ({highlighted_element.element_id})"
                )
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

    def setup(self, config: List[Dict[str, Any]], use_proxy: bool = False) -> bool:
        """
        Wrap SetupController.setup() to automatically enhance commands with app state extraction flags.

        This method intercepts the setup config, enhances Chrome/VS Code/LibreOffice commands
        with necessary CDP/UNO flags, then calls the parent setup method.
        """
        # Enhance the config with app state extraction flags
        enhanced_config = self._enhance_setup_commands(config)

        # Call parent SetupController.setup() with enhanced config
        return super().setup(enhanced_config, use_proxy)

    def _enhance_setup_commands(self, config: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhance setup commands to include necessary flags for app state extraction.

        Args:
            config: Original setup configuration list

        Returns:
            Enhanced configuration with app state extraction flags
        """
        enhanced_config = []

        for cfg in config:
            config_type = cfg["type"]
            parameters = cfg["parameters"].copy()

            # Enhance Chrome commands
            if config_type == "chrome_open_tabs":
                # Chrome setup already includes CDP flags in _chrome_open_tabs_setup
                urls = parameters.get("urls_to_open", [])
                if urls:
                    self.logger.info(
                        f"Chrome setup detected with {len(urls)} URLs - CDP flags already included"
                    )
                enhanced_config.append(cfg)

            # Enhance launch commands for VS Code and LibreOffice
            elif config_type == "launch":
                command = parameters.get("command", [])
                if isinstance(command, str):
                    command = command.split()

                enhanced_command = self._enhance_launch_command(command)
                if enhanced_command != command:
                    parameters["command"] = enhanced_command
                    self.logger.info(f"Enhanced launch command: {command} -> {enhanced_command}")

                enhanced_config.append({"type": config_type, "parameters": parameters})
            else:
                # Pass through unchanged
                enhanced_config.append(cfg)

        return enhanced_config

    def _enhance_launch_command(self, command: List[str]) -> List[str]:
        """
        Enhance a launch command with app state extraction flags if needed.

        Args:
            command: Original command list

        Returns:
            Enhanced command with necessary flags
        """
        if not command:
            return command

        enhanced_command = command.copy()
        app_name = command[0].lower()

        # VS Code enhancement
        if app_name in ["code", "vscode", "visual-studio-code"]:
            cdp_flags = ["--inspect-extensions=9229"]
            for flag in cdp_flags:
                if flag not in enhanced_command:
                    enhanced_command.append(flag)
                    self.logger.info(f"Added VS Code CDP flag: {flag}")

        # Chrome/Chromium enhancement
        elif app_name in ["google-chrome", "chrome", "chromium"]:
            cdp_flags = [
                "--remote-debugging-port=9222",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-web-security",
                "--disable-features=VizDisplayCompositor",
                "--user-data-dir=/tmp/chrome-debug",
            ]
            for flag in cdp_flags:
                if flag not in enhanced_command:
                    enhanced_command.append(flag)
                    self.logger.info(f"Added Chrome CDP flag: {flag}")

        # LibreOffice enhancement
        elif app_name in ["libreoffice", "soffice"]:
            uno_flags = [
                "--accept=socket,host=localhost,port=2002;urp;StarOffice.ServiceManager",
                "--headless",
            ]
            for flag in uno_flags:
                if flag not in enhanced_command:
                    enhanced_command.append(flag)
                    self.logger.info(f"Added LibreOffice UNO flag: {flag}")

        return enhanced_command

    def get_window_states(self) -> List[WindowState]:
        """Get enhanced window states using the improved extractor"""
        try:
            accessibility_tree = self.get_accessibility_tree()
            if not accessibility_tree:
                self.logger.warning("No accessibility tree available")
                return []

            # Extract window states with X11+CDP+UNO enhancement
            window_states = self._extractor.extract_window_states(accessibility_tree)

            if window_states:
                self.logger.info(f"Extracted {len(window_states)} enhanced window states")
                return window_states
            else:
                self.logger.warning("No window states extracted")
                return []

        except Exception as e:
            self.logger.error(f"Error getting enhanced window states: {e}")
            return []

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
        elif "gnome-shell" in app_name_lower:
            return "desktop"
        else:
            return "unknown"

    def _get_app_properties(self, app_type: str) -> Dict[str, Any]:
        """Get application-specific properties"""
        try:
            # Initialize autoglm_v tool classes
            tools = {
                "code": CodeTools,
                "chrome": BrowserTools,
                "vlc": VLCTools,
                "libreoffice_calc": CalcTools,
                "libreoffice_writer": WriterTools,
                "libreoffice_impress": ImpressTools,
            }

            if app_type in tools:
                tool_class = tools[app_type]

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

    def get_window_z_order(self) -> List[str]:
        """Get actual window stacking order from window manager"""
        python_code = """
import subprocess
import re

try:
    result = subprocess.run(
        ["xprop", "-root", "_NET_CLIENT_LIST_STACKING"],
        capture_output=True, text=True, timeout=2
    )

    if result.returncode != 0:
        print("FAILED")
        exit(1)

    # Parse: _NET_CLIENT_LIST_STACKING(WINDOW): window id # 0x3400001, 0x3400002, ...
    window_ids = re.findall(r"0x[0-9a-f]+", result.stdout)

    # List is bottom-to-top, so reverse for top-to-bottom
    window_ids = list(reversed(window_ids))

    for wid in window_ids:
        print(wid)

except Exception as e:
    print(f"ERROR: {e}")
    exit(1)
"""
        result = self.execute_python_command(python_code)
        if not result or result.get("status") != "success":
            raise RuntimeError(f"Failed to get window z-order from VM: {result}")

        output = result.get("output", "").strip()
        if output.startswith("ERROR"):
            raise RuntimeError(f"VM execution failed: {output}")
        if output.startswith("FAILED"):
            raise RuntimeError("VM command failed to execute")

        window_ids = [line.strip() for line in output.split("\n") if line.strip()]
        return window_ids

    def get_window_geometry(self, window_id: str) -> Optional[Dict[str, int]]:
        """Get window position and size from xwininfo"""
        python_code = f"""
import subprocess
import json

try:
    result = subprocess.run(
        ["xwininfo", "-id", "{window_id}", "-stats"],
        capture_output=True, text=True, timeout=2
    )

    if result.returncode != 0:
        print("FAILED")
        exit(1)

    geometry = {{}}
    for line in result.stdout.split("\\n"):
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

    print(json.dumps(geometry))

except Exception as e:
    print(f"ERROR: {{e}}")
    exit(1)
"""
        result = self.execute_python_command(python_code)
        if not result or result.get("status") != "success":
            raise RuntimeError(f"Failed to get window geometry from VM: {result}")

        output = result.get("output", "").strip()
        if output.startswith("ERROR"):
            raise RuntimeError(f"VM execution failed: {output}")
        if output.startswith("FAILED"):
            raise RuntimeError("VM command failed to execute")

        try:
            import json

            geometry = json.loads(output)
            return geometry if geometry else None
        except json.JSONDecodeError as e:
            raise e

    def get_window_name(self, window_id: str) -> str:
        """Get window title"""
        python_code = f"""
import subprocess

try:
    result = subprocess.run(
        ["xdotool", "getwindowname", "{window_id}"],
        capture_output=True, text=True, timeout=2
    )

    if result.returncode == 0:
        print(result.stdout.strip())
    else:
        print("FAILED")
        exit(1)

except Exception as e:
    print(f"ERROR: {{e}}")
    exit(1)
"""
        result = self.execute_python_command(python_code)
        if not result or result.get("status") != "success":
            raise RuntimeError(f"Failed to get window name from VM: {result}")

        output = result.get("output", "").strip()
        if output.startswith("ERROR"):
            raise RuntimeError(f"VM execution failed: {output}")
        if output.startswith("FAILED"):
            raise RuntimeError("VM command failed to execute")

        return output

    def get_focused_window(self) -> Optional[str]:
        """Get currently focused window ID"""
        python_code = """
import subprocess

try:
    result = subprocess.run(
        ["xdotool", "getactivewindow"],
        capture_output=True, text=True, timeout=2
    )

    if result.returncode == 0:
        window_id = result.stdout.strip()
        if window_id.isdigit():
            print(f"0x{int(window_id):x}")
        else:
            print("None")
    else:
        print("FAILED")
        exit(1)

except Exception as e:
    print(f"ERROR: {{e}}")
    exit(1)
"""
        result = self.execute_python_command(python_code)
        if not result or result.get("status") != "success":
            raise RuntimeError(f"Failed to get focused window from VM: {result}")

        output = result.get("output", "").strip()
        if output.startswith("ERROR"):
            raise RuntimeError(f"VM execution failed: {output}")
        if output.startswith("FAILED"):
            raise RuntimeError("VM command failed to execute")

        return output if output != "None" else None

    def get_current_desktop(self) -> int:
        """Get current virtual desktop"""
        python_code = """
import subprocess
import re

try:
    result = subprocess.run(
        ["xprop", "-root", "_NET_CURRENT_DESKTOP"],
        capture_output=True, text=True, timeout=2
    )

    if result.returncode == 0:
        match = re.search(r"= (\\d+)", result.stdout)
        if match:
            print(int(match.group(1)))
        else:
            print(0)
    else:
        print("FAILED")
        exit(1)

except Exception as e:
    print(f"ERROR: {{e}}")
    exit(1)
"""
        result = self.execute_python_command(python_code)
        if not result or result.get("status") != "success":
            raise RuntimeError(f"Failed to get current desktop from VM: {result}")

        output = result.get("output", "").strip()
        if output.startswith("ERROR"):
            raise RuntimeError(f"VM execution failed: {output}")
        if output.startswith("FAILED"):
            raise RuntimeError("VM command failed to execute")

        try:
            return int(output)
        except ValueError as e:
            raise e

    def get_window_desktop(self, window_id: str) -> int:
        """Get which desktop window is on"""
        python_code = f"""
import subprocess
import re

try:
    result = subprocess.run(
        ["xprop", "-id", "{window_id}", "_NET_WM_DESKTOP"],
        capture_output=True, text=True, timeout=2
    )

    if result.returncode == 0:
        match = re.search(r"= (\\d+)", result.stdout)
        if match:
            print(int(match.group(1)))
        else:
            print(-1)
    else:
        print("FAILED")
        exit(1)

except Exception as e:
    print(f"ERROR: {{e}}")
    exit(1)
"""
        result = self.execute_python_command(python_code)
        if not result or result.get("status") != "success":
            raise RuntimeError(f"Failed to get window desktop from VM: {result}")

        output = result.get("output", "").strip()
        if output.startswith("ERROR"):
            raise RuntimeError(f"VM execution failed: {output}")
        if output.startswith("FAILED"):
            raise RuntimeError("VM command failed to execute")

        try:
            return int(output)
        except ValueError as e:
            raise e

    def find_windows_for_app(self, app_name: str) -> List[str]:
        """Find all X11 window IDs for an application"""
        python_code = f"""
import subprocess

try:
    # Try by class name first
    result = subprocess.run(
        ["xdotool", "search", "--class", "{app_name}"],
        capture_output=True, text=True, timeout=2
    )

    if result.returncode == 0 and result.stdout.strip():
        window_ids = [f"0x{{int(wid):x}}" for wid in result.stdout.strip().split("\\n") if wid]
        for wid in window_ids:
            print(wid)
        exit(0)

    # Fallback: try by name
    result = subprocess.run(
        ["xdotool", "search", "--name", "{app_name}"],
        capture_output=True, text=True, timeout=2
    )

    if result.returncode == 0 and result.stdout.strip():
        window_ids = [f"0x{{int(wid):x}}" for wid in result.stdout.strip().split("\\n") if wid]
        for wid in window_ids:
            print(wid)
    else:
        print("NONE")

except Exception as e:
    print(f"ERROR: {{e}}")
    exit(1)
"""
        result = self.execute_python_command(python_code)
        if not result or result.get("status") != "success":
            raise RuntimeError(f"Failed to find windows for app from VM: {result}")

        output = result.get("output", "").strip()
        if output.startswith("ERROR"):
            raise RuntimeError(f"VM execution failed: {output}")
        if output == "NONE":
            return []

        window_ids = [line.strip() for line in output.split("\n") if line.strip()]
        return window_ids

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
