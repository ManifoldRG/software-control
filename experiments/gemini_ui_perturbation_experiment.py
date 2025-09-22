import time

import dotenv
from google import genai
from google.genai import types
from playwright.sync_api import sync_playwright

dotenv.load_dotenv()
client = genai.Client()  # it reads the API key from the environment variable GEMINI_API_KEY


def extract_nav_html(page):
    interactable_html = ""
    # Selectors for common interactable elements
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
        elements = page.query_selector_all(sel)
        for el in elements:
            # Filter visible elements only
            visible = page.evaluate(
                "el => window.getComputedStyle(el).display !== 'none' && window.getComputedStyle(el).visibility !== 'hidden'",
                el,
            )
            if visible:
                interactable_html += el.evaluate("el => el.outerHTML") + "\n"
    return interactable_html


def get_js_code_with_gemini(nav_html: str) -> str:
    # reorder_items_prompt = f"""
    # Here is the website html:
    # {nav_html}

    # Now, you are required to reorder all visible interactable UI elements within each section in a different way that is realistic by generating the javascript code for page.evaluate() argument in the below python command below:

    # Please generate **only the JavaScript code** that will go **inside the argument of page.evaluate()**

    # Output strictly only the JavaScript code inside the parentheses, nothing else. Do not include the page.evaluate wrapper, any explanation, or extra text.

    # """

    randomize_ui_design_prompt = f"""
    Here is the website html:
    {nav_html}

    Now, you are required to add 10 random UI elements to the website page within each section in a realistic and consistent way referring to the existing UI elements

    Please generate **only the JavaScript code** that will go **inside the argument of page.evaluate()**

    Output strictly only the JavaScript code inside the parentheses, nothing else. Do not include the page.evaluate wrapper, any explanation, or extra text.

    Specifically,
    - The output must be a single JavaScript immediately-invoked function expression (IIFE) or a function expression.
    - Add 10 random UI elements to each major section of the provided HTML in a realistic and consistent manner, respecting existing UI styles.
    - Ensure safety by checking that all selected DOM elements exist before manipulating them, using null checks.
    - Do NOT include the `page.evaluate` wrapper or any explanatory text; output strictly only the JavaScript code inside the parentheses.
    - Use `const` or `let` for variable declarations.
    - Do not include async code or external resource loading.
    - The code should run safely without throwing errors if elements are missing.
    """

    response = client.models.generate_content(
        model="gemini-1.5-flash-8b",
        contents=randomize_ui_design_prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0)  # Disables thinking
        ),
    )

    return response.text


def main():
    with sync_playwright() as p:
        start_time = time.time()
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto("https://www.drugs.com")
        page.screenshot(path="original_page.png")

        nav_html = extract_nav_html(page)
        end_time = time.time()
        print(f"Time taken to extract nav html: {end_time - start_time} seconds")

        start_time = time.time()
        response = get_js_code_with_gemini(nav_html)

        print("gemini response:\n", response)
        js_code = response.split("```")[1].removeprefix("javascript")

        page.evaluate(js_code)
        page.screenshot(path="reordered_page.png")
        end_time = time.time()
        print("page updated with: \n", js_code)
        print(f"Time taken to update page: {end_time - start_time} seconds")
        browser.close()


if __name__ == "__main__":
    main()
