"""
PerturbationController: Execute perturbation code
Clean interface for VM manipulation
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from playwright.sync_api import Page, sync_playwright

from OSWorld.desktop_env.controllers.python import PythonController
from OSWorld.desktop_env.controllers.setup import SetupController
from perturbation_engine.pipeline.data_models import WindowState
from perturbation_engine.tools.app_state_manager import AppStateExtractor

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


class PerturbationSetupController(SetupController):
    """Setup controller with perturbation enhancements"""

    def __init__(
        self, vm_ip: str, server_port: int, chromium_port: int = 9222, client_password: str = "", **kwargs
    ):
        # Ensure logging is configured for subprocess (only if not already configured)
        if not logging.getLogger().handlers:
            from perturbation_engine.configure_logging import configure_logging

            configure_logging()

        # Separate kwargs for SetupController
        setup_kwargs = {
            k: v for k, v in kwargs.items() if k in ["vlc_port", "cache_dir", "screen_width", "screen_height"]
        }

        SetupController.__init__(
            self, vm_ip, server_port, chromium_port, client_password=client_password, **setup_kwargs
        )
        self.vm_ip = vm_ip
        self.server_port = server_port
        self.chromium_port = chromium_port
        self.client_password = client_password
        self.logger = logging.getLogger(__name__)

        # Debug logging for port configuration
        self.logger.info(f"PerturbationSetupController initialized with chromium_port: {self.chromium_port}")

    def _launch_setup(self, command: Union[str, List[str]], shell: bool = False):
        """
        Override OSWorld's _launch_setup to handle shell commands properly.

        This fixes the issue where commands like "VLC_VERBOSE=-1 vlc" get split
        incorrectly, breaking environment variable syntax.
        """
        # If command is a string and contains environment variables, force shell=True
        if isinstance(command, str) and "=" in command and not command.startswith("--"):
            self.logger.info(f"Detected environment variable in command, forcing shell=True: {command}")
            shell = True

        # Call parent method with potentially modified shell parameter
        super()._launch_setup(command, shell)

    def _chrome_open_tabs_setup(self, urls_to_open: List[str]):
        """
        Override Chrome tab opening to add proper timing for socat port forwarding.
        This fixes the Chrome connection timing issue.
        """
        host = self.vm_ip
        port = self.chromium_port
        remote_debugging_url = f"http://{host}:{port}"

        self.logger.info("Connect to Chrome @: %s", remote_debugging_url)

        # Wait for socat port forwarding to be ready
        self.logger.info("⏳ Waiting for socat port forwarding to be ready...")
        time.sleep(3)  # Give socat time to establish port forwarding

        # Check if port forwarding is working
        try:
            import requests

            response = requests.get(f"{remote_debugging_url}/json", timeout=2)
            if response.status_code == 200:
                self.logger.info("✅ Port forwarding is ready!")
            else:
                self.logger.warning("⚠️ Port forwarding may not be ready yet, continuing anyway...")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not verify port forwarding readiness: {e}, continuing anyway...")

        # Now call the original Chrome tab opening logic
        super()._chrome_open_tabs_setup(urls_to_open)


class PerturbationPythonController(PythonController):
    """Python controller with perturbation enhancements"""

    def __init__(
        self, vm_ip: str, server_port: int, chromium_port: int = 9222, client_password: str = "", **kwargs
    ):
        # Ensure logging is configured for subprocess (only if not already configured)
        if not logging.getLogger().handlers:
            from perturbation_engine.configure_logging import configure_logging

            configure_logging()

        # Separate kwargs for PythonController
        python_kwargs = {k: v for k, v in kwargs.items() if k in ["pkgs_prefix"]}

        PythonController.__init__(self, vm_ip, server_port, **python_kwargs)
        self.vm_ip = vm_ip
        self.server_port = server_port
        self.chromium_port = chromium_port
        self.client_password = client_password
        self.logger = logging.getLogger(__name__)

        # Debug logging for port configuration
        self.logger.info(f"PerturbationPythonController initialized with chromium_port: {self.chromium_port}")

        # Playwright connection
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

        self._extractor = AppStateExtractor(controller=self)
        self._setup_accessibility()


class PerturbationController(PerturbationPythonController, PerturbationSetupController):
    """Execute perturbation code with clean interface - combines both controllers"""

    def __init__(
        self, vm_ip: str, server_port: int, chromium_port: int = 9222, client_password: str = "", **kwargs
    ):
        # Initialize both parent controllers
        PerturbationPythonController.__init__(
            self, vm_ip, server_port, chromium_port, client_password, **kwargs
        )
        PerturbationSetupController.__init__(
            self, vm_ip, server_port, chromium_port, client_password, **kwargs
        )

        self.logger.info(f"PerturbationController initialized with chromium_port: {self.chromium_port}")

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
        """Execute perturbation with enhanced command parsing"""
        try:
            # Parse and extract the actual command from LLM-generated wrapper
            parsed_api_call, parsed_code, parsed_parameters = self._parse_llm_generated_command(
                generated_code, api_call, parameters
            )

            # Validate inputs
            if not parsed_api_call or not parsed_code:
                self.logger.warning(f"Empty parsed api_call or code: {parsed_api_call}, {parsed_code}")
                return ManipulationResult(
                    success=False,
                    operation_type=perturbation_type,
                    target_app=parameters.get("target_app", "unknown"),
                    result_data={"api_call": parsed_api_call, "generated_code": parsed_code},
                    error_message="Empty parsed api_call or generated_code",
                )

            # Validate api_call is supported
            if not self._is_api_call_supported(parsed_api_call):
                self.logger.warning(f"Unsupported api_call: {parsed_api_call}")
                return ManipulationResult(
                    success=False,
                    operation_type=perturbation_type,
                    target_app=parameters.get("target_app", "unknown"),
                    result_data={"api_call": parsed_api_call, "generated_code": parsed_code},
                    error_message=f"Unsupported api_call: {parsed_api_call}",
                )

            # Validate command syntax
            is_valid, error_msg = self._validate_command_syntax(parsed_code, parsed_api_call)
            if not is_valid:
                self.logger.warning(f"Command syntax validation failed: {error_msg}")
                return ManipulationResult(
                    success=False,
                    operation_type=perturbation_type,
                    target_app=parameters.get("target_app", "unknown"),
                    result_data={"api_call": parsed_api_call, "generated_code": parsed_code},
                    error_message=f"Command syntax validation failed: {error_msg}",
                )

            success = False
            result_data = {}

            # Core execution methods using parsed values
            if parsed_api_call == "execute_js_on_page":
                success = self.execute_js_on_page(parsed_code)
                result_data = {"api_call": parsed_api_call, "code": parsed_code}
            elif parsed_api_call == "execute_bash_command":
                success = self.execute_bash_command(parsed_code)
                result_data = {"api_call": parsed_api_call, "command": parsed_code}
            elif parsed_api_call == "execute_python_command":
                result = self.execute_python_command(parsed_code)
                success = result.get("status") == "success"
                result_data = {"api_call": parsed_api_call, "result": result}
            elif parsed_api_call == "execute_uno_command":
                result = self.execute_uno_command(parsed_code, parsed_parameters)
                success = result.get("status") == "success" and result.get("returncode", -1) == 0
                result_data = {"api_call": parsed_api_call, "code": parsed_code, "result": result}

            # Visual manipulation operations
            elif parsed_api_call == "execute_css_injection":
                success = self.execute_css_injection(parsed_code, parsed_parameters)
                result_data = {
                    "api_call": parsed_api_call,
                    "css": parsed_code,
                    "parameters": parsed_parameters,
                }
            elif parsed_api_call == "execute_dom_modification":
                success = self.execute_dom_modification(parsed_code, parsed_parameters)
                result_data = {
                    "api_call": parsed_api_call,
                    "dom_code": parsed_code,
                    "parameters": parsed_parameters,
                }
            elif parsed_api_call == "execute_theme_randomization":
                success = self.execute_theme_randomization(parsed_parameters)
                result_data = {"api_call": parsed_api_call, "parameters": parsed_parameters}
            elif parsed_api_call == "execute_layout_perturbation":
                success = self.execute_layout_perturbation(parsed_parameters)
                result_data = {"api_call": parsed_api_call, "parameters": parsed_parameters}
            elif parsed_api_call == "execute_typography_randomization":
                success = self.execute_typography_randomization(parsed_parameters)
                result_data = {"api_call": parsed_api_call, "parameters": parsed_parameters}
            elif parsed_api_call == "execute_animation_effects":
                success = self.execute_animation_effects(parsed_code, parsed_parameters)
                result_data = {
                    "api_call": parsed_api_call,
                    "animation_code": parsed_code,
                    "parameters": parsed_parameters,
                }
            elif parsed_api_call == "execute_accessibility_perturbation":
                success = self.execute_accessibility_perturbation(parsed_parameters)
                result_data = {"api_call": parsed_api_call, "parameters": parsed_parameters}

            # Freeform operations
            elif parsed_api_call == "execute_python_execution":
                success = self.execute_python_execution(parsed_code, parsed_parameters)
                result_data = {
                    "api_call": parsed_api_call,
                    "python_code": parsed_code,
                    "parameters": parsed_parameters,
                }
            elif parsed_api_call == "execute_javascript_injection":
                success = self.execute_javascript_injection(parsed_code, parsed_parameters)
                result_data = {
                    "api_call": parsed_api_call,
                    "js_code": parsed_code,
                    "parameters": parsed_parameters,
                }
            elif parsed_api_call == "execute_bash_automation":
                success = self.execute_bash_automation(parsed_code, parsed_parameters)
                result_data = {
                    "api_call": parsed_api_call,
                    "bash_code": parsed_code,
                    "parameters": parsed_parameters,
                }
            elif parsed_api_call == "execute_playwright_automation":
                success = self.execute_playwright_automation(parsed_code, parsed_parameters)
                result_data = {
                    "api_call": parsed_api_call,
                    "playwright_code": parsed_code,
                    "parameters": parsed_parameters,
                }
            elif parsed_api_call == "execute_file_system_manipulation":
                success = self.execute_file_system_manipulation(parsed_parameters)
                result_data = {"api_call": parsed_api_call, "parameters": parsed_parameters}
            elif parsed_api_call == "execute_network_perturbation":
                success = self.execute_network_perturbation(parsed_parameters)
                result_data = {"api_call": parsed_api_call, "parameters": parsed_parameters}
            elif parsed_api_call == "execute_system_integration":
                success = self.execute_system_integration(parsed_parameters)
                result_data = {"api_call": parsed_api_call, "parameters": parsed_parameters}

            # Legacy operations
            elif parsed_api_call == "manipulate_app_state":
                success = self._manipulate_app_state(parsed_parameters)
                result_data = {"api_call": parsed_api_call, "parameters": parsed_parameters}
            elif parsed_api_call == "execute_system_perturbation":
                system_type = parsed_parameters.get("system_type", "desktop_theme")
                success = self.execute_system_perturbation(system_type, parsed_parameters)
                result_data = {
                    "api_call": parsed_api_call,
                    "system_type": system_type,
                    "parameters": parsed_parameters,
                }

            # New concrete visual operations
            elif parsed_api_call == "execute_vlc_visual_effects":
                success = self.execute_vlc_visual_effects(parsed_code, parsed_parameters)
                result_data = {
                    "api_call": parsed_api_call,
                    "vlc_code": parsed_code,
                    "parameters": parsed_parameters,
                }
            elif parsed_api_call == "execute_chrome_visual_manipulation":
                success = self.execute_chrome_visual_manipulation(parsed_code, parsed_parameters)
                result_data = {
                    "api_call": parsed_api_call,
                    "chrome_code": parsed_code,
                    "parameters": parsed_parameters,
                }
            elif parsed_api_call == "execute_libreoffice_visual_formatting":
                success = self.execute_libreoffice_visual_formatting(parsed_code, parsed_parameters)
                result_data = {
                    "api_call": parsed_api_call,
                    "libreoffice_code": parsed_code,
                    "parameters": parsed_parameters,
                }
            elif parsed_api_call == "execute_system_theme_coherence":
                success = self.execute_system_theme_coherence(parsed_parameters)
                result_data = {"api_call": parsed_api_call, "parameters": parsed_parameters}
            else:
                self.logger.error(f"Unsupported API call: {parsed_api_call}")
                return self._execute_fallback_perturbation(perturbation_type, parsed_parameters)

            # Extract detailed error message if available
            error_message = None
            if not success:
                if parsed_api_call == "execute_uno_command" and "result" in result_data:
                    result = result_data["result"]
                    if result.get("error"):
                        error_message = f"UNO command failed: {result['error']}"
                    elif result.get("returncode", 0) != 0:
                        error_message = f"UNO command failed with return code {result.get('returncode', -1)}"
                    else:
                        error_message = f"UNO command failed: {result.get('output', 'Unknown error')}"
                elif parsed_api_call == "execute_python_command" and "result" in result_data:
                    result = result_data["result"]
                    if result.get("error"):
                        error_message = f"Python command failed: {result['error']}"
                    else:
                        error_message = f"Python command failed: {result.get('output', 'Unknown error')}"
                else:
                    error_message = f"Failed to execute {parsed_api_call}"

            return ManipulationResult(
                success=success,
                operation_type=perturbation_type,
                target_app=parsed_parameters.get("target_app", "unknown"),
                result_data=result_data,
                error_message=error_message,
            )

        except Exception as e:
            self.logger.error(f"Error executing perturbation: {e}")
            return ManipulationResult(
                success=False,
                operation_type=perturbation_type,
                target_app=parameters.get("target_app", "unknown"),
                result_data={"api_call": api_call, "generated_code": generated_code},
                error_message=f"Execution error: {e}",
            )

    def _validate_command_syntax(self, command: str, api_call: str) -> Tuple[bool, str]:
        """Simple command validation"""
        if not command or not command.strip():
            return False, "Empty command"

        if api_call == "execute_bash_command":
            # Basic bash validation
            if command.count('"') % 2 != 0:
                return False, "Unclosed quotes in bash command"
            if command.count("'") % 2 != 0:
                return False, "Unclosed quotes in bash command"

        elif api_call == "execute_python_command":
            # Basic Python validation
            try:
                compile(command, "<string>", "exec")
            except SyntaxError as e:
                return False, f"Python syntax error: {e}"

        elif api_call == "execute_css_injection":
            # Basic CSS validation
            if "{" in command and "}" not in command:
                return False, "Unclosed CSS braces"

        return True, ""

    def _is_api_call_supported(self, api_call: str) -> bool:
        supported_calls = [
            "execute_js_on_page",
            "execute_bash_command",
            "execute_python_command",
            "execute_uno_command",
            "execute_css_injection",
            "execute_dom_modification",
            "execute_theme_randomization",
            "execute_layout_perturbation",
            "execute_typography_randomization",
            "execute_animation_effects",
            "execute_accessibility_perturbation",
            "execute_python_execution",
            "execute_javascript_injection",
            "execute_bash_automation",
            "execute_playwright_automation",
            "execute_file_system_manipulation",
            "execute_network_perturbation",
            "execute_system_integration",
            "manipulate_app_state",
            "execute_system_perturbation",
            # New concrete visual operations
            "execute_vlc_visual_effects",
            "execute_chrome_visual_manipulation",
            "execute_libreoffice_visual_formatting",
            "execute_system_theme_coherence",
        ]
        return api_call in supported_calls

    def execute_vlc_visual_effects(self, vlc_code: str, parameters: Dict[str, Any]) -> bool:
        """Execute VLC-specific visual effects"""
        try:
            # Parse VLC operation from generated code
            if "apply_video_filter" in vlc_code:
                if "blur" in vlc_code:
                    command = 'VLCTools.set_settings("video-filter", "blur")'
                elif "sepia" in vlc_code:
                    command = 'VLCTools.set_settings("video-filter", "sepia")'
                elif "invert" in vlc_code:
                    command = 'VLCTools.set_settings("video-filter", "invert")'
                else:
                    command = 'VLCTools.set_settings("video-filter", "blur")'
            elif "change_aspect_ratio" in vlc_code:
                if "4_3" in vlc_code:
                    command = 'VLCTools.set_settings("aspect-ratio", "4:3")'
                elif "16_9" in vlc_code:
                    command = 'VLCTools.set_settings("aspect-ratio", "16:9")'
                else:
                    command = 'VLCTools.set_settings("aspect-ratio", "16:9")'
            elif "modify_video_brightness" in vlc_code:
                command = 'VLCTools.set_settings("brightness", "1.2")'
            else:
                # Default VLC theme change
                command = 'VLCTools.set_settings("qt-theme", "dark")'

            result = self.execute_python_command(command)
            return result.get("status") == "success"
        except Exception as e:
            self.logger.error(f"Error executing VLC visual effects: {e}")
            return False

    def execute_chrome_visual_manipulation(self, chrome_code: str, parameters: Dict[str, Any]) -> bool:
        """Execute Chrome-specific visual manipulation"""
        try:
            # Parse Chrome operation from generated code
            if "inject_custom_css" in chrome_code:
                if "red_theme" in chrome_code:
                    css = "body { background-color: #ff0000 !important; color: #ffffff !important; }"
                elif "dark_theme" in chrome_code:
                    css = "body { background-color: #1a1a1a !important; color: #ffffff !important; }"
                elif "high_contrast" in chrome_code:
                    css = "body { background-color: #000000 !important; color: #ffff00 !important; }"
                else:
                    css = "body { background-color: #ff0000 !important; }"

                return self.execute_css_injection(css, parameters)
            elif "modify_page_colors" in chrome_code:
                css = "body { filter: hue-rotate(180deg) !important; }"
                return self.execute_css_injection(css, parameters)
            else:
                # Default Chrome theme change
                return self.execute_css_injection(
                    "body { background-color: #ff0000 !important; }", parameters
                )
        except Exception as e:
            self.logger.error(f"Error executing Chrome visual manipulation: {e}")
            return False

    def execute_libreoffice_visual_formatting(
        self, libreoffice_code: str, parameters: Dict[str, Any]
    ) -> bool:
        """Execute LibreOffice-specific visual formatting"""
        try:
            # Parse LibreOffice operation from generated code
            if "randomize_cell_colors" in libreoffice_code:
                uno_code = 'CalcTools.format_range("A1:C10", "background_color", "#ff0000")'
            elif "change_font_rendering" in libreoffice_code:
                uno_code = 'CalcTools.set_font("Arial", 14)'
            elif "modify_border_styles" in libreoffice_code:
                uno_code = 'CalcTools.format_range("A1:C10", "border_style", "thick")'
            else:
                # Default LibreOffice theme change
                uno_code = 'CalcTools.set_theme("dark")'

            result = self.execute_uno_command(uno_code, parameters)
            return result.get("status") == "success"
        except Exception as e:
            self.logger.error(f"Error executing LibreOffice visual formatting: {e}")
            return False

    def execute_system_theme_coherence(self, parameters: Dict[str, Any]) -> bool:
        """Execute system-level theme changes for coherence"""
        try:
            target_app = parameters.get("target_app", "system")
            commands = self._get_coherent_system_commands(target_app)

            # Execute all commands
            success = True
            for command in commands:
                if not self.execute_bash_command(command):
                    success = False

            return success
        except Exception as e:
            self.logger.error(f"Error executing system theme coherence: {e}")
            return False

    def _get_coherent_system_commands(self, target_app: str) -> List[str]:
        """Get coherent system commands based on target app"""
        app_commands = {
            "vlc": [
                'gsettings set org.gnome.desktop.interface gtk-theme "Adwaita-dark"',
                'gsettings set org.gnome.desktop.background picture-uri "file:///usr/share/backgrounds/ubuntu-mate-photos/ubuntu-mate-dark.jpg"',
            ],
            "chrome": [
                'gsettings set org.gnome.desktop.interface gtk-theme "Adwaita-dark"',
                'gsettings set org.gnome.desktop.interface font-name "Liberation Sans 14"',
            ],
            "google_chrome": [
                'gsettings set org.gnome.desktop.interface gtk-theme "Adwaita-dark"',
                'gsettings set org.gnome.desktop.interface font-name "Liberation Sans 14"',
            ],
            "libreoffice_calc": [
                'gsettings set org.gnome.desktop.interface gtk-theme "Adwaita-dark"',
                'gsettings set org.gnome.desktop.interface cursor-theme "Adwaita"',
            ],
            "libreoffice_writer": [
                'gsettings set org.gnome.desktop.interface gtk-theme "Adwaita-dark"',
                'gsettings set org.gnome.desktop.interface cursor-theme "Adwaita"',
            ],
        }

        return app_commands.get(
            target_app.lower(), ['gsettings set org.gnome.desktop.interface gtk-theme "Adwaita-dark"']
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

        Uses execute_python_command with subprocess for simple commands to avoid
        the _append_event bug in the VM server's run_bash_script endpoint.
        Falls back to run_bash_script for complex shell operations.

        Checks BOTH status=="success" AND returncode==0 for true success.
        """
        try:
            # Clean up the command if it contains markdown
            if "```" in command:
                command = command.split("```")[1].removeprefix("bash").strip()

            # For simple commands, use Python subprocess to avoid VM server bugs
            if self._is_simple_command(command):
                python_code = f"""
import subprocess
import sys

try:
    result = subprocess.run(
        ['bash', '-c', {repr(command)}],
        capture_output=True,
        text=True,
        timeout=30
    )

    if result.returncode == 0:
        print("SUCCESS")
        if result.stdout:
            print(f"STDOUT: {{result.stdout}}")
    else:
        print("FAILED")
        if result.stderr:
            print(f"STDERR: {{result.stderr}}")
        sys.exit(result.returncode)

except subprocess.TimeoutExpired:
    print("TIMEOUT")
    sys.exit(124)
except Exception as e:
    print(f"ERROR: {{e}}")
    sys.exit(1)
"""
                result = self.execute_python_command(python_code)

                if result and result.get("status") == "success":
                    output = result.get("output", "")
                    if "SUCCESS" in output:
                        self.logger.info(f"Bash command executed successfully: {command}")
                        return True
                    else:
                        self.logger.warning(f"Bash command failed: {command}")
                        self.logger.warning(f"Output: {output}")
                        return False
                else:
                    self.logger.warning(f"Python subprocess execution failed: {result}")
                    return False
            else:
                # For complex commands, use run_bash_script (may fail due to VM server bug)
                result = self.run_bash_script(command, timeout=30)

                # Check both status and return code
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

    def _parse_llm_generated_command(
        self, generated_code: str, api_call: str, parameters: Dict[str, Any]
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Parse LLM-generated commands that may be wrapped in function calls.

        Handles various formats:
        1. Direct commands: 'gsettings set theme' -> ('execute_bash_command', 'gsettings set theme', {})
        2. Function wrappers: 'execute_bash_command(\'gsettings set theme\')' -> ('execute_bash_command', 'gsettings set theme', {})
        3. CSS with parameters: 'execute_css_injection(\'body {color: red}\', {\'target_app\': \'chrome\'})' -> ('execute_css_injection', 'body {color: red}', {'target_app': 'chrome'})
        4. UNO with parameters: 'execute_uno_command(\'CalcTools.set_theme("dark")\', {\'target_app\': \'calc\'})' -> ('execute_uno_command', 'CalcTools.set_theme("dark")', {'target_app': 'calc'})

        Returns: (parsed_api_call, parsed_code, parsed_parameters)
        """
        import json
        import re

        if not generated_code:
            return api_call, "", parameters

        # Pattern to match function calls with parameters
        # Matches: function_name('arg1', {'param': 'value'}) or function_name('arg1')
        function_pattern = r"(\w+)\(['\"](.*?)['\"](?:,\s*(\{.*?\}))?\)"
        match = re.search(function_pattern, generated_code, re.DOTALL)

        if match:
            function_name = match.group(1)
            code_content = match.group(2)
            params_str = match.group(3)

            # Parse parameters if present
            parsed_params = parameters.copy()
            if params_str:
                try:
                    # Clean up the parameters string
                    params_str = params_str.strip()
                    # Convert single quotes to double quotes for valid JSON
                    params_str = params_str.replace("'", '"')
                    parsed_params.update(json.loads(params_str))
                except (json.JSONDecodeError, ValueError) as e:
                    self.logger.warning(f"Failed to parse parameters '{params_str}': {e}")

            # Validate that the function name matches a supported API call
            if self._is_api_call_supported(function_name):
                self.logger.debug(f"Parsed LLM command: {function_name} -> {code_content[:100]}...")
                return function_name, code_content, parsed_params
            else:
                self.logger.warning(f"Unsupported function name in generated code: {function_name}")
                # Fall back to original api_call
                return api_call, code_content, parsed_params

        # If no function wrapper found, treat as direct command
        self.logger.debug(f"No function wrapper found, using direct command: {generated_code[:100]}...")
        return api_call, generated_code, parameters

    def _is_simple_command(self, command: str) -> bool:
        """
        Determine if a command is simple enough to use Python subprocess instead of run_bash_script.
        Simple commands don't use complex shell features like pipes, redirects, conditionals, etc.
        """
        # Commands that should use Python subprocess (avoid VM server bug)
        simple_patterns = [
            "gsettings",
            "notify-send",
            "mkdir",
            "touch",
            "rm ",
            "cp ",
            "mv ",
            "ls ",
            "pwd",
            "echo ",
            "cat ",
            "head ",
            "tail ",
            "grep ",
            "find ",
            "which ",
            "ps ",
            "kill ",
            "pkill ",
            "wmctrl ",
            "xdotool ",
            "xprop ",
            "xwininfo ",
        ]

        # Check if command contains complex shell features
        complex_features = ["|", "&&", "||", ">", ">>", "<", "<<", ";", "(", ")", "$(", "`", "\\$"]
        has_complex_features = any(feature in command for feature in complex_features)

        # Check if it's a simple command
        is_simple = any(pattern in command for pattern in simple_patterns)

        return is_simple and not has_complex_features

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
        """Get Playwright page with simplified connection management"""
        if self._page is not None:
            return self._page

        # Use the configured chromium_port (9222) which is forwarded to Chrome's internal port 1337
        remote_debugging_url = f"http://{self.vm_ip}:{self.chromium_port}"
        self.logger.info(f"Connecting to Chrome at {remote_debugging_url}")

        # Connection logic with better logging and timing
        for attempt in range(5):
            try:
                self.logger.info(f"Connection attempt {attempt + 1}/5: Starting Playwright...")
                self._playwright = sync_playwright().start()

                self.logger.info(
                    f"Connection attempt {attempt + 1}/5: Connecting to Chrome at {remote_debugging_url}..."
                )
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

                self.logger.info(
                    f"✅ Successfully connected to Chrome at {remote_debugging_url} on attempt {attempt + 1}"
                )
                return self._page

            except Exception as e:
                if attempt < 4:
                    self.logger.warning(f"❌ Connection attempt {attempt + 1}/5 failed: {e}")
                    self.logger.info(f"🔄 Retrying in 3 seconds... (attempt {attempt + 2}/5)")
                    # Clean up partial connection
                    self._cleanup_playwright_connection()
                    time.sleep(3)
                else:
                    self.logger.error(f"❌ Failed to connect to Chrome after 5 attempts. Last error: {e}")
                    self._cleanup_playwright_connection()
                    break

        # If connection failed, try launching Chrome
        self.logger.info("🚀 All connection attempts failed, attempting to launch Chrome...")
        return self._launch_chrome_and_connect()

    def _check_chrome_readiness(self, remote_debugging_url: str) -> bool:
        """
        Check if Chrome is ready to accept connections by testing the debugging endpoint.
        """
        try:
            import requests

            response = requests.get(f"{remote_debugging_url}/json", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def _cleanup_playwright_connection(self):
        """Clean up Playwright connection resources"""
        try:
            if self._playwright:
                self._playwright.stop()
        except Exception:
            pass
        finally:
            self._playwright = None
            self._browser = None
            self._context = None
            self._page = None

    def _kill_existing_chrome(self):
        """Kill existing Chrome processes to ensure clean launch"""
        try:
            import platform

            # Determine Chrome executable names
            chrome_names = ["google-chrome", "chrome", "chromium"]
            if platform.system() == "Windows":
                chrome_names = ["chrome.exe", "chromium.exe"]

            # Kill Chrome processes
            for chrome_name in chrome_names:
                python_code = f"""
import subprocess
import signal
import os

try:
    # Find Chrome processes
    result = subprocess.run(['pgrep', '-f', '{chrome_name}'],
                           capture_output=True, text=True, timeout=5)

    if result.returncode == 0 and result.stdout.strip():
        pids = result.stdout.strip().split('\\n')
        for pid in pids:
            if pid.strip():
                try:
                    os.kill(int(pid), signal.SIGTERM)
                    print(f"Killed Chrome process {{pid}}")
                except (ValueError, ProcessLookupError, PermissionError) as e:
                    print(f"Could not kill process {{pid}}: {{e}}")

        # Wait a bit for processes to terminate
        import time
        time.sleep(2)

        # Force kill if still running
        result = subprocess.run(['pgrep', '-f', '{chrome_name}'],
                               capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\\n')
            for pid in pids:
                if pid.strip():
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                        print(f"Force killed Chrome process {{pid}}")
                    except (ValueError, ProcessLookupError, PermissionError) as e:
                        print(f"Could not force kill process {{pid}}: {{e}}")
    else:
        print("No Chrome processes found")

except Exception as e:
    print(f"Error killing Chrome processes: {{e}}")
"""
                result = self.execute_python_command(python_code)
                if result and result.get("status") == "success":
                    self.logger.info(f"Chrome cleanup completed: {result.get('output', '')}")
                else:
                    self.logger.warning(f"Chrome cleanup failed: {result}")

        except Exception as e:
            self.logger.error(f"Error in Chrome cleanup: {e}")

    def _launch_chrome_and_connect(self) -> Optional[Page]:
        """Launch Chrome with simplified connection logic"""
        try:
            import json
            import platform

            import requests

            # Determine Chrome executable based on architecture
            app = "chromium" if "arm" in platform.machine() else "google-chrome"
            command = [
                app,
                "--remote-debugging-port=1337",  # Chrome runs on 1337, socat forwards 9222 to this
                "--no-first-run",
                "--disable-web-security",
                "--user-data-dir=/tmp/chrome-debug",
            ]

            self.logger.info(f"Launching Chrome with command: {' '.join(command)}")

            # Launch Chrome via VM server
            payload = json.dumps({"command": command, "shell": False})
            headers = {"Content-Type": "application/json"}
            backend_url = f"http://{self.vm_ip}:{self.server_port}"

            response = requests.post(f"{backend_url}/setup/launch", headers=headers, data=payload, timeout=30)
            if response.status_code != 200:
                self.logger.error(f"Failed to launch Chrome: {response.status_code} - {response.text}")
                return None

            # Wait for Chrome to start
            self.logger.info("⏳ Waiting 5 seconds for Chrome to fully start...")
            time.sleep(5)

            # Try to connect with simplified logic
            remote_debugging_url = f"http://{self.vm_ip}:{self.chromium_port}"
            self.logger.info(f"🔗 Attempting to connect to launched Chrome at {remote_debugging_url}...")
            self._playwright = sync_playwright().start()

            # Single connection attempt with better error handling
            try:
                self._browser = self._playwright.chromium.connect_over_cdp(remote_debugging_url)
                self.logger.info("✅ Successfully connected to launched Chrome")
            except Exception as e:
                self.logger.error(f"❌ Failed to connect to launched Chrome: {e}")
                self.logger.info("💡 This might be due to Chrome still starting up or port forwarding issues")
                self._cleanup_playwright_connection()
                return None

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

            self.logger.info(f"Successfully launched and connected to Chrome at {remote_debugging_url}")
            return self._page

        except Exception as e:
            self.logger.error(f"Failed to launch Chrome and connect: {e}")
            self._cleanup_playwright_connection()
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
        the _append_event bug in the VM server's bash script execution.

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
        Setup method that delegates to parent SetupController without any modifications.
        This preserves OSWorld's original setup flow completely.
        """
        # Delegate directly to parent SetupController.setup() without any modifications
        return super().setup(config, use_proxy)

    def _launch_setup(self, command: Union[str, List[str]], shell: bool = False):
        """
        Override OSWorld's _launch_setup to handle shell commands properly.

        This fixes the issue where commands like "VLC_VERBOSE=-1 vlc" get split
        incorrectly, breaking environment variable syntax.
        """
        # If command is a string and contains environment variables, force shell=True
        if isinstance(command, str) and "=" in command and not command.startswith("--"):
            self.logger.info(f"Detected environment variable in command, forcing shell=True: {command}")
            shell = True

        # Call parent method with potentially modified shell parameter
        super()._launch_setup(command, shell)

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

    def _open_setup(self, path: str):
        """
        Override _open_setup to handle LibreOffice files with proper flags to avoid recovery mode.
        For LibreOffice files, use specific commands with --norestore flag.
        For other files, delegate to parent implementation.
        """
        if not path:
            raise Exception(f"Setup Open - Invalid path ({path}).")

        # Check if this is a LibreOffice file
        libreoffice_extensions = {".xlsx", ".xls", ".ods", ".docx", ".doc", ".odt", ".pptx", ".ppt", ".odp"}
        file_extension = os.path.splitext(path.lower())[1]

        if file_extension in libreoffice_extensions:
            # Use LibreOffice-specific command with --norestore flag
            self.logger.info(f"Using LibreOffice-specific opening for {path} (extension: {file_extension})")
            self._open_libreoffice_file(path, file_extension)
        else:
            # Delegate to parent implementation for non-LibreOffice files
            super()._open_setup(path)

    def _open_libreoffice_file(self, path: str, file_extension: str):
        """
        Open LibreOffice files with proper flags to avoid recovery mode.

        Args:
            path: Path to the LibreOffice file
            file_extension: File extension (e.g., '.xlsx', '.docx')
        """
        try:
            # Map file extensions to LibreOffice applications
            app_mapping = {
                ".xlsx": "calc",
                ".xls": "calc",
                ".ods": "calc",
                ".docx": "writer",
                ".doc": "writer",
                ".odt": "writer",
                ".pptx": "impress",
                ".ppt": "impress",
                ".odp": "impress",
            }

            app_name = app_mapping.get(file_extension, "calc")

            # Build LibreOffice command with --norestore flag
            command = [
                "libreoffice",
                "--norestore",  # Prevent recovery mode
                "--nodefault",  # Don't open default document
                f"--{app_name}",  # Specify the application type
                path,
            ]

            self.logger.info(f"Opening LibreOffice file with command: {' '.join(command)}")

            # Use the launch setup method to execute the command
            self._launch_setup(command)

            # Wait a bit for LibreOffice to start
            time.sleep(3)

            self.logger.info(f"LibreOffice {app_name} opened successfully for {path}")

        except Exception as e:
            self.logger.error(f"Failed to open LibreOffice file {path}: {e}")
            # Fallback to parent implementation if LibreOffice-specific opening fails
            self.logger.info("Falling back to default file opening method")
            super()._open_setup(path)

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
