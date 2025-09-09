"""Enhanced controller that extends OSWorld's PythonController with Playwright page access"""

import logging
from typing import Any, Dict, Optional

from playwright.sync_api import Page, sync_playwright

from OSWorld.desktop_env.controllers.python import PythonController
from perturbation_engine.control.gemini_controller import GeminiWebPageController
from perturbation_engine.data_types import PerturbationSpec, PerturbationType


class PerturbationController(PythonController):
    """Extended PythonController that provides Playwright page access"""

    def __init__(self, vm_ip: str, server_port: int, chromium_port: int = 9222, **kwargs):
        super().__init__(vm_ip, server_port, **kwargs)
        self.chromium_port = chromium_port
        self.logger = logging.getLogger(__name__)
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self.vlm_controller = GeminiWebPageController()

    def _ensure_playwright_connection(self) -> bool:
        """Ensure Playwright connection to Chrome is established"""
        if self._page is not None:
            return True

        try:
            self._playwright = sync_playwright().start()
            remote_debugging_url = f"http://{self.vm_ip}:{self.chromium_port}"

            # Connect to existing Chrome instance
            self._browser = self._playwright.chromium.connect_over_cdp(remote_debugging_url)

            # Get the first context (should be the only one)
            if self._browser.contexts:
                self._context = self._browser.contexts[0]
                # Get the first page (active tab)
                if self._context.pages:
                    self._page = self._context.pages[0]
                else:
                    # Create a new page if none exists
                    self._page = self._context.new_page()
            else:
                # Create new context if none exists
                self._context = self._browser.new_context()
                self._page = self._context.new_page()

            self.logger.info(f"Connected to Chrome via Playwright at {remote_debugging_url}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to connect to Chrome via Playwright: {e}")
            return False

    @property
    def page(self) -> Optional[Page]:
        """Get the current Playwright page"""
        if self._ensure_playwright_connection():
            return self._page
        return None

    def can_handle(self, perturbation_type: PerturbationType) -> bool:
        return perturbation_type in [PerturbationType.UI_VISUAL, PerturbationType.VISUAL_DISTRACTOR]

    def apply_perturbation(self, spec: PerturbationSpec, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply UI component injection using Gemini"""
        try:
            nav_html = self.get_interactable_html()
            if not nav_html:
                return {"applied": False, "error": "Could not extract page HTML"}

            js_code = self.vlm_controller.get_ui_perturbation_js_code(nav_html, spec.parameters)
            if not js_code:
                return {"applied": False, "error": "Could not generate JavaScript code"}

            success = self.execute_js_on_page(js_code)

            return {
                "applied": success,
                "method": "gemini_ui_perturbation",
                "js_code": js_code,
                "parameters": spec.parameters,
            }

        except Exception as e:
            self.logger.error(f"Error applying VLM perturbation: {e}")
            return {"applied": False, "error": str(e)}

    def execute_js_on_page(self, js_code: str) -> Any:
        """Execute JavaScript code on the current page"""
        try:
            # Clean up the JavaScript code (remove markdown formatting)
            if "```" in js_code:
                js_code = js_code.split("```")[1].removeprefix("javascript").strip()

            self.page.evaluate(js_code)
            self.logger.info(f"Executed JavaScript code: {js_code[:100]}...")
            return True

        except Exception as e:
            self.logger.error(f"Error executing JavaScript code: {e}")
            return False

    def get_page_html(self, selector: str = None) -> str:
        """Get HTML content from the current page"""
        page = self.page
        if not page:
            return ""

        try:
            if selector:
                element = page.query_selector(selector)
                return element.inner_html() if element else ""
            else:
                return page.content()
        except Exception as e:
            self.logger.error(f"Error getting page HTML: {e}")
            return ""

    def get_interactable_html(self) -> str:
        """Extract interactable HTML elements from the current page"""
        try:
            interactable_html = ""
            # Common interactable elements
            selectors = [
                "a",
                "button",
                "input",
                "select",
                "textarea",
                "[tabindex]",
                "[role=button]",
                "[role=link]",
                "[contenteditable=true]",
            ]

            for sel in selectors:
                elements = self.page.query_selector_all(sel)
                for el in elements:
                    # Filter visible elements only
                    visible = self.page.evaluate(
                        "el => window.getComputedStyle(el).display !== 'none' && window.getComputedStyle(el).visibility !== 'hidden'",
                        el,
                    )
                    if visible:
                        interactable_html += el.evaluate("el => el.outerHTML") + "\n"

            return interactable_html

        except Exception as e:
            self.logger.error(f"Error extracting HTML: {e}")
            return ""

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
