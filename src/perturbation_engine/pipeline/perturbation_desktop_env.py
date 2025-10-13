"""
PerturbationDesktopEnv: Extended env with chrome management
Clean interface for environment management
"""

import logging
import os
from typing import Tuple

from OSWorld.desktop_env.desktop_env import DesktopEnv
from perturbation_engine.control.perturbation_controller import (
    PerturbationController,
    PerturbationSetupController,
)


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
        """Override to use PerturbationController instead of PythonController"""
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

            self.controller = PerturbationController(
                vm_ip=self.vm_ip,
                server_port=self.server_port,
                chromium_port=self.chromium_port,
                client_password=self.client_password,
                vlc_port=self.vlc_port,
                cache_dir=self.cache_dir_base,
                screen_width=self.screen_width,
                screen_height=self.screen_height,
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
        if hasattr(self.controller, "close_playwright"):
            self.controller.close_playwright()
        super().close()

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
        """Get current timestamp"""
        import datetime

        return datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
