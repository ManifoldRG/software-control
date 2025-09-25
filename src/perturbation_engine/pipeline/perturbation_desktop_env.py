import logging
import os
from typing import Tuple

from OSWorld.desktop_env.desktop_env import DesktopEnv
from perturbation_engine.control.perturbation_controller import PerturbationController


class PerturbationDesktopEnv(DesktopEnv):
    """Enhanced DesktopEnv that provides perturbation controller"""

    def __init__(
        self,
        provider_name: str = "vmware",
        region: str = None,
        path_to_vm: str = None,
        snapshot_name: str = "init_state",
        action_space: str = "pyautogui",
        cache_dir: str = "cache",
        screen_size: Tuple[int] = (
            int(os.environ.get("SCREEN_WIDTH", 1920)),
            int(os.environ.get("SCREEN_HEIGHT", 1080)),
        ),
        headless: bool = False,
        require_a11y_tree: bool = True,
        require_terminal: bool = False,
        os_type: str = "Ubuntu",
        enable_proxy: bool = False,
        client_password: str = "",
        chromium_port: int = 9222,
    ):
        self.logger = logging.getLogger(__name__)
        super().__init__(
            provider_name=provider_name,
            region=region,
            path_to_vm=path_to_vm,
            snapshot_name=snapshot_name,
            action_space=action_space,
            cache_dir=cache_dir,
            screen_size=screen_size,
            headless=headless,
            require_a11y_tree=require_a11y_tree,
            require_terminal=require_terminal,
            os_type=os_type,
            enable_proxy=enable_proxy,
            client_password=client_password,
        )

        self.logger.info("Perturbation controller initialized")

    def _start_emulator(self):
        """Override to use PerturbationController instead of PythonController"""
        super()._start_emulator()

        # Replace the controller with our enhanced version
        self.controller = PerturbationController(
            vm_ip=self.vm_ip, server_port=self.server_port, chromium_port=self.chromium_port
        )
        self.logger.info("Replaced controller with PerturbationController")

    def close(self) -> None:
        """Close both the perturbation controller and original environment"""
        self.controller.close_playwright()
        super().close()

    def _get_obs(self):
        """Get comprehensive observation including DOM, A11Y, and app-specific state"""
        try:
            # Get basic observation
            obs = {
                "screenshot": self.controller.get_screenshot(),
                "accessibility_tree": self.controller.get_accessibility_tree()
                if self.require_a11y_tree
                else None,
                "terminal": self.controller.get_terminal_output() if self.require_terminal else None,
                "app_info": self.controller.get_app_info(),
                "instruction": self.instruction,
                "timestamp": self._get_timestamp(),
            }

            # Add DOM tree if available (for browser apps)
            if hasattr(self.controller, "get_page_html"):
                try:
                    obs["dom_tree"] = self.controller.get_page_html()
                except Exception as e:
                    self.logger.warning(f"Could not get DOM tree: {e}")
                    obs["dom_tree"] = None

            # Add A11Y tree if available
            if hasattr(self.controller, "get_accessibility_tree"):
                try:
                    obs["a11y_tree"] = self.controller.get_accessibility_tree()
                except Exception as e:
                    self.logger.warning(f"Could not get A11Y tree: {e}")
                    obs["a11y_tree"] = None

            # Add browser-specific state
            if hasattr(self.controller, "page") and self.controller.page:
                try:
                    obs["page_title"] = self.controller.page.title()
                    obs["url"] = self.controller.page.url
                    obs["viewport_size"] = self.controller.page.viewport_size
                except Exception as e:
                    self.logger.warning(f"Could not get browser state: {e}")

            # Extract app-specific state based on current task
            obs.update(self._extract_app_specific_state())

            # Add OS-level state
            obs.update(self._get_os_level_state())

            return obs

        except Exception as e:
            self.logger.error(f"Error getting observation: {e}")
            return {
                "screenshot": None,
                "accessibility_tree": None,
                "terminal": None,
                "app_info": None,
                "instruction": self.instruction,
                "timestamp": self._get_timestamp(),
            }

    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        import datetime

        return datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

    def _extract_app_specific_state(self) -> dict:
        """Extract app-specific state based on current task type"""
        try:
            # Get task type from instruction or config
            task_type = self._detect_task_type_from_instruction()

            app_state = {}

            if task_type == "chrome":
                app_state = self._extract_browser_state()
            elif task_type in ["libreoffice_calc", "libreoffice_writer", "libreoffice_impress"]:
                app_state = self._extract_libreoffice_state(task_type)
            elif task_type == "gimp":
                app_state = self._extract_gimp_state()
            elif task_type == "vs_code":
                app_state = self._extract_vscode_state()
            elif task_type == "os":
                app_state = self._extract_os_state()
            elif task_type == "thunderbird":
                app_state = self._extract_thunderbird_state()
            elif task_type == "vlc":
                app_state = self._extract_vlc_state()
            else:
                app_state = self._extract_generic_app_state()

            return app_state

        except Exception as e:
            self.logger.warning(f"Could not extract app-specific state: {e}")
            return {}

    def _detect_task_type_from_instruction(self) -> str:
        """Detect task type from instruction or environment"""
        try:
            # Check if we have a task config
            if hasattr(self, "task_config") and self.task_config:
                return self.task_config.get("task_type", "chrome")

            # Try to detect from instruction
            if hasattr(self, "instruction") and self.instruction:
                instruction_lower = self.instruction.lower()
                if "chrome" in instruction_lower or "browser" in instruction_lower:
                    return "chrome"
                elif "calc" in instruction_lower or "spreadsheet" in instruction_lower:
                    return "libreoffice_calc"
                elif "writer" in instruction_lower or "document" in instruction_lower:
                    return "libreoffice_writer"
                elif "impress" in instruction_lower or "presentation" in instruction_lower:
                    return "libreoffice_impress"
                elif "gimp" in instruction_lower or "image" in instruction_lower:
                    return "gimp"
                elif "code" in instruction_lower or "vscode" in instruction_lower:
                    return "vs_code"
                elif "file" in instruction_lower or "folder" in instruction_lower:
                    return "os"
                elif "email" in instruction_lower or "thunderbird" in instruction_lower:
                    return "thunderbird"
                elif "video" in instruction_lower or "vlc" in instruction_lower:
                    return "vlc"

            # Default to chrome
            return "chrome"

        except Exception as e:
            self.logger.warning(f"Could not detect task type: {e}")
            return "chrome"

    def _extract_browser_state(self) -> dict:
        """Extract browser-specific state using Playwright APIs"""
        try:
            browser_state = {}

            if hasattr(self.controller, "page") and self.controller.page:
                try:
                    browser_state.update(
                        {
                            "page_title": self.controller.page.title(),
                            "url": self.controller.page.url,
                            "viewport_size": self.controller.page.viewport_size,
                            "app_type": "browser",
                        }
                    )
                except Exception as e:
                    self.logger.warning(f"Could not get browser page state: {e}")

            return browser_state

        except Exception as e:
            self.logger.warning(f"Could not extract browser state: {e}")
            return {"app_type": "browser"}

    def _extract_libreoffice_state(self, task_type: str) -> dict:
        """Extract LibreOffice-specific state using UNO API"""
        try:
            libreoffice_state = {"app_type": "libreoffice"}

            # Try to get LibreOffice state via terminal commands
            if hasattr(self.controller, "get_terminal_output"):
                try:
                    # Get active document info
                    terminal_output = self.controller.get_terminal_output()
                    if terminal_output:
                        libreoffice_state["terminal_output"] = terminal_output
                except Exception as e:
                    self.logger.warning(f"Could not get LibreOffice terminal output: {e}")

            # Add task-specific state
            if task_type == "libreoffice_calc":
                libreoffice_state["document_type"] = "spreadsheet"
            elif task_type == "libreoffice_writer":
                libreoffice_state["document_type"] = "document"
            elif task_type == "libreoffice_impress":
                libreoffice_state["document_type"] = "presentation"

            return libreoffice_state

        except Exception as e:
            self.logger.warning(f"Could not extract LibreOffice state: {e}")
            return {"app_type": "libreoffice"}

    def _extract_gimp_state(self) -> dict:
        """Extract GIMP-specific state using Python-Fu API"""
        try:
            gimp_state = {"app_type": "gimp"}

            # Try to get GIMP state via terminal commands
            if hasattr(self.controller, "get_terminal_output"):
                try:
                    terminal_output = self.controller.get_terminal_output()
                    if terminal_output:
                        gimp_state["terminal_output"] = terminal_output
                except Exception as e:
                    self.logger.warning(f"Could not get GIMP terminal output: {e}")

            return gimp_state

        except Exception as e:
            self.logger.warning(f"Could not extract GIMP state: {e}")
            return {"app_type": "gimp"}

    def _extract_vscode_state(self) -> dict:
        """Extract VS Code-specific state"""
        try:
            vscode_state = {"app_type": "vs_code"}

            # Try to get VS Code state via terminal commands
            if hasattr(self.controller, "get_terminal_output"):
                try:
                    terminal_output = self.controller.get_terminal_output()
                    if terminal_output:
                        vscode_state["terminal_output"] = terminal_output
                except Exception as e:
                    self.logger.warning(f"Could not get VS Code terminal output: {e}")

            return vscode_state

        except Exception as e:
            self.logger.warning(f"Could not extract VS Code state: {e}")
            return {"app_type": "vs_code"}

    def _extract_os_state(self) -> dict:
        """Extract OS-level state"""
        try:
            os_state = {"app_type": "os"}

            # Get OS-level information
            if hasattr(self.controller, "get_terminal_output"):
                try:
                    terminal_output = self.controller.get_terminal_output()
                    if terminal_output:
                        os_state["terminal_output"] = terminal_output
                except Exception as e:
                    self.logger.warning(f"Could not get OS terminal output: {e}")

            return os_state

        except Exception as e:
            self.logger.warning(f"Could not extract OS state: {e}")
            return {"app_type": "os"}

    def _extract_thunderbird_state(self) -> dict:
        """Extract Thunderbird-specific state"""
        try:
            thunderbird_state = {"app_type": "thunderbird"}

            # Try to get Thunderbird state via terminal commands
            if hasattr(self.controller, "get_terminal_output"):
                try:
                    terminal_output = self.controller.get_terminal_output()
                    if terminal_output:
                        thunderbird_state["terminal_output"] = terminal_output
                except Exception as e:
                    self.logger.warning(f"Could not get Thunderbird terminal output: {e}")

            return thunderbird_state

        except Exception as e:
            self.logger.warning(f"Could not extract Thunderbird state: {e}")
            return {"app_type": "thunderbird"}

    def _extract_vlc_state(self) -> dict:
        """Extract VLC-specific state"""
        try:
            vlc_state = {"app_type": "vlc"}

            # Try to get VLC state via terminal commands
            if hasattr(self.controller, "get_terminal_output"):
                try:
                    terminal_output = self.controller.get_terminal_output()
                    if terminal_output:
                        vlc_state["terminal_output"] = terminal_output
                except Exception as e:
                    self.logger.warning(f"Could not get VLC terminal output: {e}")

            return vlc_state

        except Exception as e:
            self.logger.warning(f"Could not extract VLC state: {e}")
            return {"app_type": "vlc"}

    def _extract_generic_app_state(self) -> dict:
        """Extract generic app state"""
        try:
            generic_state = {"app_type": "unknown"}

            # Try to get generic app state via terminal commands
            if hasattr(self.controller, "get_terminal_output"):
                try:
                    terminal_output = self.controller.get_terminal_output()
                    if terminal_output:
                        generic_state["terminal_output"] = terminal_output
                except Exception as e:
                    self.logger.warning(f"Could not get generic app terminal output: {e}")

            return generic_state

        except Exception as e:
            self.logger.warning(f"Could not extract generic app state: {e}")
            return {"app_type": "unknown"}

    def _get_os_level_state(self) -> dict:
        """Get OS-level state information"""
        try:
            os_state = {}

            # Get window information
            if hasattr(self.controller, "get_vm_window_size"):
                try:
                    window_info = self.controller.get_vm_window_size("")
                    os_state["window_size"] = window_info
                except Exception as e:
                    self.logger.warning(f"Could not get window size: {e}")

            # Get screen information
            if hasattr(self.controller, "get_vm_screen_size"):
                try:
                    screen_info = self.controller.get_vm_screen_size()
                    os_state["screen_size"] = screen_info
                except Exception as e:
                    self.logger.warning(f"Could not get screen size: {e}")

            # Get system information
            if hasattr(self.controller, "get_system_info"):
                try:
                    system_info = self.controller.get_system_info()
                    os_state["system_info"] = system_info
                except Exception as e:
                    self.logger.warning(f"Could not get system info: {e}")

            return os_state

        except Exception as e:
            self.logger.warning(f"Could not get OS-level state: {e}")
            return {}
