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

    def execute_perturbation(
        self, perturbation_type: str, generated_code: str, api_call: str, parameters: Dict[str, Any]
    ) -> ManipulationResult:
        """Execute perturbation using generated code with sophisticated handling"""
        try:
            success = False
            result_data = {}

            if api_call == "execute_js_on_page":
                success = self.execute_js_on_page(generated_code)
                result_data = {"api_call": api_call, "code": generated_code[:100] + "..."}
            elif api_call == "execute_bash_command":
                success = self.execute_bash_command(generated_code)
                result_data = {"api_call": api_call, "command": generated_code}
            elif api_call == "execute_python_command":
                result = self.execute_python_command(generated_code)
                success = result.get("status") == "success"
                result_data = {"api_call": api_call, "result": result}
            elif api_call == "execute_uno_command":
                success = self.execute_uno_command(generated_code, parameters)
                result_data = {"api_call": api_call, "code": generated_code[:100] + "..."}
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
        """Execute bash command with improved error handling"""
        try:
            # Clean up the command if it contains markdown
            if "```" in command:
                command = command.split("```")[1].removeprefix("bash").strip()

            # Execute with proper error handling
            result = self.execute_python_command(
                f"import subprocess; result = subprocess.run(['bash', '-c', '{command}'], capture_output=True, text=True); print(f'STDOUT: {{result.stdout}}'); print(f'STDERR: {{result.stderr}}'); print(f'Return code: {{result.returncode}}')"
            )
            success = result.get("status") == "success"
            if success:
                self.logger.info(f"Bash command executed: {command}")
            else:
                self.logger.warning(f"Bash command failed: {command}")
            return success
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

    def execute_uno_command(self, uno_code: str, parameters: Dict[str, Any]) -> bool:
        """Execute UNO command for LibreOffice manipulation"""
        try:
            # Clean up the UNO code
            if "```" in uno_code:
                uno_code = uno_code.split("```")[1].removeprefix("python").strip()

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
    {uno_code}

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
    raise
"""

            result = self.execute_python_command(python_wrapper)
            return result.get("status") == "success"

        except Exception as e:
            self.logger.error(f"Error executing UNO command: {e}")
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
