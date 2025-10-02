"""
PerturbationDesktopEnv: Extended env with chrome management
Clean interface for environment management
"""

import logging
import os
from enum import Enum
from typing import Any, Dict, List, Tuple

from OSWorld.desktop_env.desktop_env import DesktopEnv
from perturbation_engine.control.perturbation_controller import PerturbationController


class AppType(Enum):
    """Application types"""

    BROWSER = "browser"
    LIBREOFFICE_CALC = "libreoffice_calc"
    LIBREOFFICE_WRITER = "libreoffice_writer"
    LIBREOFFICE_IMPRESS = "libreoffice_impress"
    VS_CODE = "vs_code"
    GIMP = "gimp"
    VLC = "vlc"
    THUNDERBIRD = "thunderbird"
    FILE_MANAGER = "file_manager"
    TERMINAL = "terminal"
    UNKNOWN = "unknown"


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
        # Ensure logging is configured for subprocess (only if not already configured)
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
        super()._start_emulator()

        # Replace the controller with our enhanced version
        self.controller = PerturbationController(
            vm_ip=self.vm_ip, server_port=self.server_port, chromium_port=self.chromium_port
        )
        self.logger.info("Replaced controller with PerturbationController")

        # Ensure AT-SPI accessibility is enabled for proper state extraction
        # Do this in a non-blocking way to avoid setup hangs
        try:
            self.logger.info("Setting up AT-SPI accessibility for app state extraction...")
            accessibility_ok = self.controller.ensure_accessibility_enabled()
            if accessibility_ok:
                self.logger.info("AT-SPI accessibility enabled successfully")
            else:
                self.logger.warning(
                    "AT-SPI accessibility may not be fully enabled - will retry later if needed"
                )
        except Exception as e:
            self.logger.warning(f"AT-SPI setup encountered error (non-fatal): {e}")
            # Continue anyway - accessibility can be enabled later if needed

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
                "app_states": self.get_app_states_from_accessibility_tree(),
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
                "app_states": [],
                "instruction": self.instruction,
                "timestamp": self._get_timestamp(),
                "url": "",
                "window_size": {},
            }

    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        import datetime

        return datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

    def get_app_states_from_accessibility_tree(self) -> List[Dict[str, Any]]:
        """
        Extract app states for LLM prompting.

        Delegates to AppStateExtractor in PerturbationController with fallback handling.
        """
        try:
            return self._extract_app_states_with_fallback()
        except Exception as e:
            self.logger.error(f"Error extracting app states: {e}")
            return self._create_empty_app_state()

    def _extract_app_states_with_fallback(self) -> List[Dict[str, Any]]:
        """Extract app states with proper fallback handling."""
        if hasattr(self.controller, "get_comprehensive_app_states"):
            return self.controller.get_comprehensive_app_states()

        self.logger.warning("Controller doesn't have app state extractor, returning empty states")
        return self._create_empty_app_state()

    def _create_empty_app_state(self) -> List[Dict[str, Any]]:
        """Create empty app state for fallback scenarios."""
        return [
            {
                "app_type": "unknown",
                "app_name": "unknown",
                "current_view": "unknown",
                "key_elements": [],
                "task_context": "No accessible applications detected",
                "element_count": 0,
            }
        ]
