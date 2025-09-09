import logging
import os
from typing import Any, Dict

from google import genai
from google.genai import types


class GeminiWebPageController:
    """Simple VLM controller for UI perturbations using Gemini"""

    def __init__(self, api_key: str = None):
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = None

        if self.api_key:
            self.client = genai.Client()  # Uses API key from environment
        else:
            self.logger.error("Gemini API not available or API key not provided")

    def get_ui_perturbation_js_code(self, nav_html: str, parameters: Dict[str, Any]) -> str:
        """Generate JavaScript code using Gemini (following experiment pattern)"""
        if not self.client:
            return ""

        try:
            # Get parameters
            action = parameters.get("action", "ui_injection")
            # num_components = parameters.get("num_components", 10)

            if action == "theme_change":
                prompt = self._create_theme_change_prompt(nav_html, parameters)
            elif action == "add_success_indicators":
                prompt = self._create_success_indicators_prompt(nav_html, parameters)
            else:
                prompt = self._create_ui_injection_prompt(nav_html, parameters)

            response = self.client.models.generate_content(
                model="gemini-1.5-flash-8b",
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0)  # Disables thinking
                ),
            )

            return response.text

        except Exception as e:
            self.logger.error(f"Error generating JavaScript with Gemini: {e}")
            return ""

    def _create_ui_injection_prompt(self, nav_html: str, parameters: Dict[str, Any]) -> str:
        """Create prompt for UI injection (following experiment pattern)"""
        num_components = parameters.get("num_components", 10)

        return f"""
        Here is the website html:
        {nav_html}

        Now, you are required to add {num_components} random UI elements similar to the existing UI elements to the website page within each section in a realistic and consistent way

        Please generate **only the JavaScript code** that will go **inside the argument of page.evaluate()**

        Output strictly only the JavaScript code inside the parentheses, nothing else. Do not include the page.evaluate wrapper, any explanation, or extra text.

        Specifically,
        - The output must be a single JavaScript immediately-invoked function expression (IIFE) or a function expression.
        - Add {num_components} random UI elements to each major section of the provided HTML in a realistic and consistent manner, respecting existing UI styles.
        - Ensure safety by checking that all selected DOM elements exist before manipulating them, using null checks.
        - Do NOT include the `page.evaluate` wrapper or any explanatory text; output strictly only the JavaScript code inside the parentheses.
        - Use `const` or `let` for variable declarations.
        - Do not include async code or external resource loading.
        - The code should run safely without throwing errors if elements are missing.
        - Keep the line of code as short as possible.
        """

    def _create_theme_change_prompt(self, nav_html: str, parameters: Dict[str, Any]) -> str:
        """Create prompt for theme change"""
        theme = parameters.get("theme", "dark")

        return f"""
        Here is the website html:
        {nav_html}

        Now, you are required to change the theme of the website to {theme} theme.

        Please generate **only the JavaScript code** that will go **inside the argument of page.evaluate()**

        Output strictly only the JavaScript code inside the parentheses, nothing else. Do not include the page.evaluate wrapper, any explanation, or extra text.

        Specifically,
        - Apply {theme} theme styling to the page
        - Modify background colors, text colors, and other visual elements
        - Ensure the changes are consistent and realistic
        - Use `const` or `let` for variable declarations.
        - Do not include async code or external resource loading.
        - The code should run safely without throwing errors if elements are missing.
        """

    def _create_success_indicators_prompt(self, nav_html: str, parameters: Dict[str, Any]) -> str:
        """Create prompt for success indicators"""
        indicators = parameters.get("indicators", ["checkmark", "progress_bar"])

        return f"""
        Here is the website html:
        {nav_html}

        Now, you are required to add success indicators to the website page.

        Indicators to add: {", ".join(indicators)}

        Please generate **only the JavaScript code** that will go **inside the argument of page.evaluate()**

        Output strictly only the JavaScript code inside the parentheses, nothing else. Do not include the page.evaluate wrapper, any explanation, or extra text.

        Specifically,
        - Add success indicators like checkmarks, progress bars, or completion messages
        - Position them appropriately on the page
        - Make them visually consistent with the existing design
        - Use `const` or `let` for variable declarations.
        - Do not include async code or external resource loading.
        - The code should run safely without throwing errors if elements are missing.
        """
