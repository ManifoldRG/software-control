"""Scene analysis components for identifying perturbable elements."""

import logging
from pathlib import Path
from timeit import default_timer as timer

from playwright.sync_api import sync_playwright

# from perturbation_engine.scene.component_grouper import ComponentGrouper
from perturbation_engine.scene.data import Element, SceneAnalysis


class SceneAnalyzer:
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        # self._component_grouper = ComponentGrouper()

    def analyze_scene(self, mhtml_file: Path, task: str) -> SceneAnalysis:
        """Analyze MHTML content and identify elements."""
        self._logger.info(f"Analyzing scene: {mhtml_file.name}\nTask: {task}")

        start_time = timer()
        elements = self._identify_elements(mhtml_file)
        end_time = timer()
        self._logger.info(f"Time taken to identify elements: {end_time - start_time:.4f} seconds")

        # Group elements into functional components
        # start_time = timer()
        # functional_components = self._component_grouper.group_elements(elements)

        # end_time = timer()
        # self._logger.info(f"Time taken to group components: {end_time - start_time:.4f} seconds")

        # FIXME: assign all interactive elements as goal relevant for now
        goal_relevant = [e for e in elements if e.is_interactive]
        background = [e for e in elements if not e.is_interactive]

        self._logger.info(
            f"Elements: total={len(elements)}, goal_relevant={len(goal_relevant)}, background={len(background)}"
        )
        # self._logger.info(f"Functional components: {len(functional_components)}")

        # if goal_relevant:
        #     self._logger.info(f"Sample goal relevant element: {goal_relevant[0].element_type}")
        # if background:
        #     self._logger.info(f"Sample background element: {background[0].element_type}")
        # if functional_components:
        #     self._logger.info(f"Sample component: {functional_components[0].component_type.value}")

        return SceneAnalysis(
            scene_id=mhtml_file.stem,  # TODO: could use a better scene id
            elements=elements,
            goal_relevant_elements=goal_relevant,
            background_elements=background,
            # functional_components=functional_components,
            plausibility_score=0.8,  # TODO: mock values
            solvability_score=0.7,  # TODO: mock values
        )

    def _identify_elements(self, mhtml_file: Path) -> list[Element]:
        """Use Playwright to identify all elements and their properties."""
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=False)
            page = browser.new_page()

            # Load the MHTML file directly - this preserves the original page structure
            page.goto(f"file://{mhtml_file.resolve()}")

            # Wait for page to load completely
            page.wait_for_load_state("networkidle")

            elements = []
            processed_elements = set()  # Track processed element IDs to avoid duplicates

            # Focus on body content only, skip head/meta elements
            body_elements = page.locator("body *").all()

            for _, element in enumerate(body_elements):
                try:
                    tag_name = element.evaluate("el => el.tagName.toLowerCase()")

                    # Skip script, style, and other non-visible elements
                    if tag_name in ["script", "style", "noscript", "meta", "link", "title", "head"]:
                        continue

                    # Check if element is actually visible
                    is_visible = element.evaluate("""
                        el => {
                            const styles = window.getComputedStyle(el);
                            const rect = el.getBoundingClientRect();

                            return (
                                styles.display !== 'none' &&
                                styles.visibility !== 'hidden' &&
                                styles.opacity !== '0' &&
                                el.offsetWidth > 0 &&
                                el.offsetHeight > 0 &&
                                rect.width > 0 &&
                                rect.height > 0
                            );
                        }
                    """)

                    if not is_visible:
                        continue

                    # Check if element is in viewport
                    is_in_viewport = element.evaluate("""
                        el => {
                            const rect = el.getBoundingClientRect();
                            return (
                                rect.top < window.innerHeight &&
                                rect.bottom > 0 &&
                                rect.left < window.innerWidth &&
                                rect.right > 0
                            );
                        }
                    """)

                    if not is_in_viewport:
                        continue

                    # Check if element has meaningful content
                    has_content = element.evaluate("""
                        el => {
                            return (
                                el.textContent?.trim().length > 0 ||
                                el.getAttribute('href') ||
                                el.getAttribute('onclick') ||
                                el.getAttribute('role') ||
                                ['button', 'input', 'a', 'select', 'textarea', 'img'].includes(el.tagName.toLowerCase())
                            );
                        }
                    """)

                    if not has_content:
                        continue

                    # Get unique element identifier to prevent duplicates
                    element_id = element.evaluate("""
                        el => {
                            // Use a combination of tag, id, and text content for uniqueness
                            const id = el.id || '';
                            const text = el.textContent?.trim().substring(0, 50) || '';
                            const href = el.getAttribute('href') || '';
                            return `${el.tagName.toLowerCase()}-${id}-${text}-${href}`;
                        }
                    """)

                    # Skip if we've already processed this element
                    if element_id in processed_elements:
                        continue

                    # Generate selector
                    if element.get_attribute("id"):
                        selector = f"#{element.get_attribute('id')}"
                    elif element.get_attribute("class"):
                        classes = element.get_attribute("class").split()
                        if classes:
                            selector = f".{classes[0]}"
                        else:
                            selector = tag_name
                    else:
                        selector = tag_name

                    # Determine if interactive - more comprehensive check
                    is_interactive = (
                        tag_name in ["button", "input", "a", "select", "textarea"]
                        or element.get_attribute("onclick") is not None
                        or element.get_attribute("role") in ["button", "link", "tab", "menuitem"]
                        or element.get_attribute("href") is not None
                        or element.get_attribute("type") in ["button", "submit", "reset"]
                        or element.get_attribute("tabindex") is not None
                    )

                    elements.append(
                        Element(
                            element_id=element_id,
                            element_type=tag_name,
                            selector=selector,
                            is_interactive=is_interactive,
                        )
                    )

                    # Track this element as processed
                    processed_elements.add(element_id)
                    page.locator(selector).highlight()

                except Exception:
                    # Skip elements that can't be processed
                    continue

            self._logger.info(f"Found {len(elements)} elements. Check console for details.")

            # Don't close browser automatically - let user close it
            return elements
