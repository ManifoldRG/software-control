"""Enhanced controller that extends OSWorld's PythonController with Playwright page access"""

import logging
from typing import Any, Dict, Optional

from playwright.sync_api import Page, sync_playwright

from OSWorld.desktop_env.controllers.python import PythonController
from OSWorld.desktop_env.controllers.setup import SetupController
from perturbation_engine.control.gemini_controller import GeminiWebPageController
from perturbation_engine.data_types import PerturbationSpec, PerturbationType


class PerturbationController(PythonController, SetupController):
    """Extended PythonController that provides Playwright page access"""

    def __init__(
        self, vm_ip: str, server_port: int, chromium_port: int = 9222, **kwargs
    ):
        PythonController.__init__(self, vm_ip, server_port, **kwargs)
        SetupController.__init__(self, vm_ip, server_port, chromium_port, **kwargs)
        self.vm_ip = vm_ip
        self.server_port = server_port
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
            self._browser = self._playwright.chromium.connect_over_cdp(
                remote_debugging_url
            )

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

            self.logger.info(
                f"Connected to Chrome via Playwright at {remote_debugging_url}"
            )
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
        return perturbation_type in [
            PerturbationType.UI_VISUAL,
            PerturbationType.VISUAL_DISTRACTOR,
        ]

    def apply_perturbation(
        self, spec: PerturbationSpec, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply UI component injection using Gemini"""
        try:
            action = spec.parameters.get("action", "ui_injection")
            menu_elements = None
            if action == "reorder_menu_elements":
                menu_elements = self.get_header_menu_elements(self.page)
                if not menu_elements or menu_elements.get("header_count", 0) == 0:
                    return {
                        "applied": False,
                        "error": "Could not extract menu elements",
                    }

            text_content = None
            if action == "rephrase_text":
                text_content = self.get_page_text_content(self.page)
                if not text_content:
                    return {"applied": False, "error": "Could not extract text content"}
                
            style_tags = self.get_style_tags()
            if not style_tags:
                return {"applied": False, "error": "Could not extract style tags"}

            nav_html = self.get_interactable_html()
            if not nav_html:
                return {"applied": False, "error": "Could not extract page HTML"}

            js_code = self.vlm_controller.get_ui_perturbation_js_code(
                nav_html,
                spec.parameters,
                menu_elements=menu_elements,
                text_elements=text_content,
                style_tags=style_tags,
            )
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
            interactable_elements = []
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
                        interactable_elements.append(el)

            # highlight all interactable elements
            for el in interactable_elements:
                el.evaluate("el => el.style.border = '2px solid red';")

            return interactable_html

        except Exception as e:
            self.logger.error(f"Error extracting HTML: {e}")
            return ""

    def get_header_menu_elements(page) -> dict:
        """Find header containers and extract interactable elements using both positional and semantic criteria"""
        try:
            result = {
                "header_containers": "",
                "menu_interactables": "",
                "header_count": 0,
            }

            # Step 1: Find likely header containers using positional criteria
            header_selectors = [
                "header",
                "nav",
                "[role='banner']",
                "[role='navigation']",
                "[class*='header']",
                "[class*='navbar']",
                "[class*='nav-bar']",
                "[class*='navigation']",
                "[class*='topbar']",
                "[class*='top-bar']",
                "[id*='header']",
                "[id*='navbar']",
                "[id*='navigation']",
                "[id*='top']",
            ]

            header_containers = []

            # Find all potential header containers
            for selector in header_selectors:
                elements = page.query_selector_all(selector)
                for el in elements:
                    is_header = page.evaluate(
                        """
                        el => {
                            const rect = el.getBoundingClientRect();
                            const styles = window.getComputedStyle(el);
                            
                            // Must be visible
                            if (styles.display === 'none' || styles.visibility === 'hidden') return false;
                            
                            // Check positioning - headers are typically at top of viewport
                            const isAtTop = rect.top < window.innerHeight * 0.3; // Top 30% of viewport
                            
                            // Check if it spans a good portion of width (typical for headers)
                            const isWideEnough = rect.width > window.innerWidth * 0.5; // At least 50% width
                            
                            // Check for fixed/sticky positioning (common for headers)
                            const isFixed = styles.position === 'fixed' || styles.position === 'sticky';
                            
                            // Check z-index (headers often have high z-index)
                            const hasHighZIndex = parseInt(styles.zIndex) > 100;
                            
                            return isAtTop || isFixed || (isWideEnough && isAtTop) || hasHighZIndex;
                        }
                    """,
                        el,
                    )

                    if is_header and el not in header_containers:
                        header_containers.append(el)

            # Step 2: Within each header container, find interactable elements
            menu_interactables = []

            for header in header_containers:

                # Get HTML of header container (element only, no children)
                header_info = header.evaluate(
                    """
                    el => {
                        const tagName = el.tagName.toLowerCase();
                        const id = el.id ? ` id="${el.id}"` : '';
                        const className = el.className ? ` class="${el.className}"` : '';
                        const role = el.getAttribute('role') ? ` role="${el.getAttribute('role')}"` : '';
                        return `<${tagName}${id}${className}${role}>`;
                    }
                """
                )
                result["header_containers"] += (
                    header_info + "\n---HEADER SEPARATOR---\n"
                )

                # Common interactable elements
                interactable_selectors = [
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

                for selector in interactable_selectors:
                    elements = header.query_selector_all(selector)
                    for el in elements:
                        is_interactable = page.evaluate(
                            """
                            el => {
                                const styles = window.getComputedStyle(el);
                                const rect = el.getBoundingClientRect();
                                
                                // Must be visible
                                
                                if (styles.display === 'none' || styles.visibility === 'hidden') return false;
                                if (rect.width === 0 || rect.height === 0) return false;
                                
                                // Must not be disabled
                                if (el.disabled) return false;
                                
                                // Check if it has meaningful content (text or accessible name)
                                const text = el.textContent.trim();
                                const ariaLabel = el.getAttribute('aria-label');
                                const title = el.getAttribute('title');
                                
                                return text.length > 0 || ariaLabel || title;
                            }
                        """,
                            el,
                        )

                        if is_interactable and el not in menu_interactables:
                            menu_interactables.append(el)

                            # Get HTML of interactable element
                            el_html = el.evaluate("el => el.outerHTML")
                            result["menu_interactables"] += el_html + "\n"

            result["header_count"] = len(header_containers)
            return result

        except Exception as e:
            print(f"Error extracting header menu elements: {e}")
            return {
                "header_containers": "",
                "menu_interactables": "",
                "header_count": 0,
            }

    def get_text_content(self) -> list:
        """Return visible text content of targeted UI elements for use in Playwright text selectors."""
        try:
            targeted_selectors = [
                "button",
                "a",
                "label",
                "input[placeholder]",
                "textarea[placeholder]",
                "[role=button]",
                "[role=link]",
                "[role=heading]",
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
            ]
            text_contents = []
            for sel in targeted_selectors:
                elements = self.page.query_selector_all(sel)
                for el in elements:
                    # Ensure the element is visible
                    visible = self.page.evaluate(
                        """el => {
                            const style = window.getComputedStyle(el);
                            const rect = el.getBoundingClientRect();
                            return (
                                style.display !== 'none' &&
                                style.visibility !== 'hidden' &&
                                rect.width > 0 &&
                                rect.height > 0
                            );
                        }""",
                        el,
                    )
                    if not visible:
                        continue
                    # Get text content suitable for playwright text selectors
                    text = el.evaluate(
                        "el => el.innerText ? el.innerText.trim() : (el.value ? el.value.trim() : (el.placeholder ? el.placeholder.trim() : ''))"
                    )
                    if text:
                        text_contents.append(text)
            # Return unique, non-empty text contents
            return list(sorted(set(filter(None, text_contents))))
        except Exception as e:
            self.logger.error(f"Error extracting text content: {e}")
            return []

    def get_style_tags(self) -> list:
        """Return style tag contents from the current page"""
        try:
            style_tags = self.page.query_selector_all("style")
            styles = [tag.evaluate("el => el.innerHTML") for tag in style_tags]
            return styles
        except Exception as e:
            self.logger.error(f"Error extracting style tags: {e}")
            return []

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
