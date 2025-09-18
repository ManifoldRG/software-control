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

    def get_ui_perturbation_js_code(
        self,
        nav_html: str,
        parameters: Dict[str, Any],
        menu_elements: Dict[str, str] = None,
        text_elements: str = None,
        style_tags: list[str] = None,
    ) -> str:
        """Generate JavaScript code using Gemini (following experiment pattern)"""
        if not self.client:
            return ""

        try:
            # Get parameters
            action = parameters.get("action", "ui_injection")
            if action == "theme_change":
                prompt = self._create_theme_change_prompt(nav_html, parameters)
            elif action == "add_success_indicators":
                prompt = self._create_success_indicators_prompt(nav_html, parameters)
            elif action == "reorder_menu_elements":
                prompt = self._reorder_menu_items_prompt(menu_elements, parameters)
            elif action == "rephrase_text":
                prompt = self._rephrase_text_prompt(text_elements, parameters)
            elif action == "change_logo":
                prompt = self._change_logo_prompt(menu_elements, parameters)
            elif action == "add_popup":
                prompt = self._add_popup_prompt(style_tags, parameters)
            else:
                prompt = self._create_ui_injection_prompt(nav_html, parameters)

            response = self.client.models.generate_content(
                model="gemini-1.5-flash-8b",
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=0
                    )  # Disables thinking
                ),
            )

            return response.text

        except Exception as e:
            self.logger.error(f"Error generating JavaScript with Gemini: {e}")
            return ""

    def _create_ui_injection_prompt(
        self, nav_html: str, parameters: Dict[str, Any]
    ) -> str:
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

    def _create_theme_change_prompt(
        self, nav_html: str, parameters: Dict[str, Any]
    ) -> str:
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

    def _create_success_indicators_prompt(
        self, nav_html: str, parameters: Dict[str, Any]
    ) -> str:
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

    def _reorder_menu_items_prompt(
        self, menu_elements: Dict[str, str], parameters: Dict[str, Any]
    ) -> str:
        """Create prompt for menu reordering"""

        return f"""
        Here are the likely menu container elements:
        {menu_elements["header_containers"]}

        Here are the likely interactable menu items:
        {menu_elements["menu_interactables"]}

        Now, you are required to reorder the menu items in a realistic way. Identify the menu container most likely to be the main navigation menu, then reorder its items.

        Please generate **only the JavaScript code** that will go **inside the argument of page.evaluate()**

        Output strictly only the JavaScript code inside the parentheses, nothing else. Do not include the page.evaluate wrapper, any explanation, or extra text.

        Specifically,
        - Identify the main navigation menu container from the provided elements
        - Reorder its items in a realistic manner
        - Preserve 
        - Make them visually consistent with the existing design
        - Use `const` or `let` for variable declarations.
        - Do not include async code or external resource loading.
        - The code should run safely without throwing errors if elements are missing.
        """

    def _rephrase_text_prompt(
        self, text_elements: list[str], parameters: Dict[str, Any]
    ) -> str:
        """Create prompt for text rephrasing"""
        num_rephrases = parameters.get("num_rephrases", 10)

        return f"""
        Here is the text content of an entire webpage:
        {"\n".join(text_elements)}

        Identify the {num_rephrases} most important text elements on the page and rephrase them, while preserving their original meaning. Then insert the rephrased text back into the page, replacing the original text.

        Please generate **only the JavaScript code** that will go **inside the argument of page.evaluate()**

        Output strictly only the JavaScript code inside the parentheses, nothing else. Do not include the page.evaluate wrapper, any explanation, or extra text.

        Specifically,
        - Identify the {num_rephrases} most important text elements on the page
        - Rephrase them, while preserving their original meaning, and keeping a similar length
        - Write JavaScript code to insert the rephrased text back into the page, replacing the original text
        - Make them visually consistent with the existing design
        - Use `const` or `let` for variable declarations.
        - Do not include async code or external resource loading.
        - The code should run safely without throwing errors if elements are missing.
        """


    def _change_logo_prompt(
        self, menu_elements: list[str], parameters: Dict[str, Any]
    ) -> str:
        """Create prompt for changing logo"""
        new_logo_url = parameters.get("new_logo", "new_logo_url")

        return f"""
        Here are the likely menu container elements:
        {menu_elements["header_containers"]}

        You need to change the logo of the website. Here is the new logo URL:
        {new_logo_url}

        Please generate **only the JavaScript code** that will go **inside the argument of page.evaluate()**

        Output strictly only the JavaScript code inside the parentheses, nothing else. Do not include the page.evaluate wrapper, any explanation, or extra text.

        Specifically,
        - Identify the logo element on the page
        - Change its source to the new logo URL
        - Make it visually consistent with the existing design
        - Use `const` or `let` for variable declarations.
        - Do not include async code or external resource loading.
        - The code should run safely without throwing errors if elements are missing.
        """

    def _add_popup_prompt(
        self, style_tag: str, parameters: Dict[str, Any]
    ) -> str:
        """Create prompt for adding a popup"""
        popup_type, popup_html = parameters.get("popup", ("info", "<div>New popup</div>"))

        return f"""
        Here is the style tag of the webpage:
        {style_tag}

        You need to add a {popup_type} popup to the website. Here is the popup HTML:
        {popup_html}

        Please generate **only the JavaScript code** that will go **inside the argument of page.evaluate()**

        Output strictly only the JavaScript code inside the parentheses, nothing else. Do not include the page.evaluate wrapper, any explanation, or extra text.

        Specifically,
        - Identify the container element on the page
        - Insert the popup HTML into the container
        - Make it visually consistent with the existing design
        - Use `const` or `let` for variable declarations.
        - Do not include async code or external resource loading.
        - The code should run safely without throwing errors if elements are missing.
        """