"""
PerturbationDesktopEnv: Extended env with chrome management
Clean interface for environment management
"""

import logging
import os
from typing import Tuple

from OSWorld.desktop_env.desktop_env import DesktopEnv
from perturbation_engine.control.perturbation_controller import (
    PerturbationPythonController,
    PerturbationSetupController,
)
from perturbation_engine.pipeline.app_state_utils import get_timestamp


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
        if not logging.getLogger().handlers:
            from perturbation_engine.configure_logging import configure_logging

            configure_logging()

        self.logger = logging.getLogger(__name__)
        self.chromium_port = chromium_port

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
        """Override to use separate perturbation controllers instead of single PerturbationController"""
        try:
            self.provider.start_emulator(self.path_to_vm, self.headless, self.os_type)
            vm_ip_ports = self.provider.get_ip_address(self.path_to_vm).split(":")
            self.vm_ip = vm_ip_ports[0]
            if len(vm_ip_ports) > 1:
                self.server_port = int(vm_ip_ports[1])
                self.chromium_port = int(vm_ip_ports[2])
                self.vnc_port = int(vm_ip_ports[3])
                self.vlc_port = int(vm_ip_ports[4])

            self.logger.info(f"PerturbationDesktopEnv using chromium_port: {self.chromium_port}")

            # Create separate controllers using composition
            self.setup_controller = PerturbationSetupController(
                vm_ip=self.vm_ip,
                server_port=self.server_port,
                chromium_port=self.chromium_port,
                client_password=self.client_password,
                vlc_port=self.vlc_port,
                cache_dir=self.cache_dir_base,
                screen_width=self.screen_width,
                screen_height=self.screen_height,
            )

            self.controller = PerturbationPythonController(
                vm_ip=self.vm_ip,
                server_port=self.server_port,
                chromium_port=self.chromium_port,
                client_password=self.client_password,
            )

        except Exception:
            try:
                self.provider.stop_emulator(self.path_to_vm)
            except Exception as stop_err:
                self.logger.warning(f"Cleanup after interrupt failed: {stop_err}")
            raise

    def mark_perturbation_applied(self):
        """Mark that a perturbation has been applied - forces reset on next trajectory"""
        self.is_environment_used = True
        self.logger.debug("Perturbation applied - environment marked as used (will reset on next trajectory)")

    def close(self) -> None:
        """Close both the perturbation controller and original environment"""
        if hasattr(self.setup_controller, "close_playwright"):
            self.setup_controller.close_playwright()
        super().close()

    # ========== Delegation Methods for Clean Interface ==========

    def get_window_states(self):
        """Get window states - delegate to python controller with setup controller"""
        return self.controller.get_window_states(setup_controller=self.setup_controller)

    def get_chrome_dom_data(self):
        """Get Chrome DOM data - delegate to setup controller"""
        return self.setup_controller.get_chrome_dom_data()

    def get_libreoffice_state(self, app_type: str = "calc"):
        """Get LibreOffice state - delegate to setup controller"""
        return self.setup_controller.get_libreoffice_state(app_type)

    def execute_perturbation(
        self, perturbation_type: str, generated_code: str, api_call: str, parameters: dict
    ):
        """Execute perturbation - delegate to python controller"""
        return self.controller.execute_perturbation(perturbation_type, generated_code, api_call, parameters)

    def start_recording(self):
        """Start recording - delegate to python controller"""
        return self.controller.start_recording()

    def end_recording(self, dest: str):
        """End recording - delegate to python controller"""
        return self.controller.end_recording(dest)

    def close_playwright(self):
        """Close Playwright - delegate to setup controller"""
        return self.setup_controller.close_playwright()

    def get_obs(self):
        """Get comprehensive observation including DOM, A11Y, and app-specific state"""
        try:
            return {
                "screenshot": self.controller.get_screenshot(),
                "accessibility_tree": self.controller.get_accessibility_tree()
                if self.require_a11y_tree
                else None,
                "terminal": self.controller.get_terminal_output() if self.require_terminal else None,
                "window_states": self.controller.get_window_states(),
                "instruction": self.instruction,
                "timestamp": self._get_timestamp(),
                "url": getattr(self.controller, "current_url", ""),
                "window_size": getattr(self.controller, "window_size", {}),
            }

        except Exception as e:
            self.logger.error(f"Error getting observation: {e}")
            return {
                "screenshot": None,
                "accessibility_tree": None,
                "terminal": None,
                "window_states": [],
                "instruction": self.instruction,
                "timestamp": self._get_timestamp(),
                "url": "",
                "window_size": {},
            }

    def _get_timestamp(self) -> str:
        """Get current timestamp - delegate to shared utility"""
        return get_timestamp()
