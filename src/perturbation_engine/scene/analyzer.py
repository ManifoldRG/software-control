"""Scene analysis components for identifying perturbable elements."""

import logging
from pathlib import Path
from timeit import default_timer as timer

from playwright.sync_api import sync_playwright

from perturbation_engine.scene.data import Element, SceneAnalysis


class SceneAnalyzer:
    def __init__(self):
        self._logger = logging.getLogger(__name__)

    def analyze_scene(self, mhtml_file: Path, task: str) -> SceneAnalysis:
        """Analyze MHTML content and identify elements."""
        self._logger.info(f"Analyzing scene: {mhtml_file.name}\nTask: {task}")

        start_time = timer()
        elements = self._identify_elements(mhtml_file)
        end_time = timer()
        self._logger.info(f"Time taken to identify elements: {end_time - start_time:.4f} seconds")

        # FIXME: assign all interactive elements as goal relevant for now
        goal_relevant = [e for e in elements if e.is_interactive]
        background = [e for e in elements if not e.is_interactive]

        self._logger.info(
            f"Elements: total={len(elements)}, goal_relevant={len(goal_relevant)}, background={len(background)}"
        )
        self._logger.info(f"Sample goal relevant element: {goal_relevant[0].element_type}")
        self._logger.info(f"Sample background element: {background[0].element_type}")

        return SceneAnalysis(
            scene_id=mhtml_file.stem,  # TODO: could use a better scene id
            elements=elements,
            goal_relevant_elements=goal_relevant,
            background_elements=background,
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

            elements = []

            # Focus on body content only, skip head/meta elements
            body_elements = page.locator("body *").all()

            for i, element in enumerate(body_elements):
                try:
                    tag_name = element.evaluate("el => el.tagName.toLowerCase()")

                    # Skip script, style, and other non-visible elements
                    if tag_name in ["script", "style", "noscript", "meta", "link", "title"]:
                        continue

                    element_id = element.get_attribute("id") or f"{tag_name}-{i + 1}"

                    # Determine if interactive - more comprehensive check
                    is_interactive = (
                        tag_name in ["button", "input", "a", "select", "textarea"]
                        or element.get_attribute("onclick") is not None
                        or element.get_attribute("role") in ["button", "link", "tab", "menuitem"]
                        or element.get_attribute("href") is not None
                        or element.get_attribute("type") in ["button", "submit", "reset"]
                    )

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

                    elements.append(
                        Element(
                            element_id=element_id,
                            element_type=tag_name,
                            selector=selector,
                            is_interactive=is_interactive,
                        )
                    )

                    # # highlight the new element in the browser
                    page.locator(selector).highlight()
                except Exception:
                    # Skip elements that can't be processed
                    continue

            browser.close()
            return elements

    def propose_configs(self, analysis: SceneAnalysis):
        """Create configurations for perturbable background elements."""
        pass
