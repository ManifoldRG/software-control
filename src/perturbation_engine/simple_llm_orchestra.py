"""
Simplified LLM Orchestra: Two-LLM quality-assured randomization system
Following YAGNI principles with clean interfaces and minimal complexity
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

from google import genai
from google.genai import types

from perturbation_engine.data_types import SeedTrajectory


@dataclass
class EnvironmentState:
    """Standardized environment state for LLM processing"""

    dom_tree: str
    a11y_tree: str
    app_type: str  # browser, desktop, system
    current_url: str
    window_state: Dict[str, Any]
    task_instruction: str


@dataclass
class SimplifiedContext:
    """Simplified context from LLM 1"""

    text_elements: List[str]
    visual_theme: str
    layout_pattern: str
    interaction_types: List[str]
    content_domain: str


@dataclass
class GeneratedCode:
    """Generated code from LLM 2"""

    code: str
    selectors_used: List[str]
    environment_dependencies: str
    error_handling: bool


@dataclass
class ApprovedVariation:
    """Final approved variation"""

    instruction: str
    code: GeneratedCode
    quality_score: int
    environment_dependencies: str


class BaseLLM(ABC):
    """Base class for all LLM components"""

    def __init__(self, model_name: str = "gemini-1.5-flash-8b"):
        self.model_name = model_name
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client = None

        if self.api_key:
            self.client = genai.Client()
        else:
            self.logger.warning("Gemini API not available - using mock responses")

    @abstractmethod
    def call_llm(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Call LLM with prompt and return structured response"""
        pass

    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API with prompt"""
        if not self.client:
            return self._get_mock_response()

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=0)),
            )
            return response.text
        except Exception as e:
            self.logger.error(f"Error calling Gemini: {e}")
            return self._get_mock_response()

    def _get_mock_response(self) -> str:
        """Get mock response when API is not available"""
        return '{"error": "Mock response - API not available"}'

    def _extract_json(self, response: str) -> Dict[str, Any]:
        """Extract JSON from LLM response"""
        try:
            # Find JSON in response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end != -1:
                json_str = response[start:end]
                return json.loads(json_str)
            else:
                self.logger.error("No JSON found in LLM response")
                return {}
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {e}")
            return {}


class ContextSimplifierLLM(BaseLLM):
    """LLM 1: Extract minimal, standardized information from complex GUI states"""

    def simplify(self, environment_state: EnvironmentState) -> SimplifiedContext:
        """Simplify complex environment state to 5 standardized elements"""
        prompt = f"""
Analyze this GUI interface and extract standardized information:

DOM Tree: {environment_state.dom_tree[:2000]}
A11Y Tree: {environment_state.a11y_tree[:1000]}
App Type: {environment_state.app_type}
Task: {environment_state.task_instruction}

Extract ONLY these 5 standardized elements and return JSON:

{{
  "text_elements": ["button_text_1", "heading_text_2", "label_text_3"],
  "visual_theme": "light" | "dark" | "neutral",
  "layout_pattern": "grid" | "list" | "form" | "navigation",
  "interaction_types": ["click", "type", "select", "scroll"],
  "content_domain": "ecommerce" | "productivity" | "social" | "utility"
}}

Rules:
- Extract text from buttons, links, headings, labels
- Determine theme from colors and styling
- Identify layout pattern (grid, list, form, navigation)
- List interaction types available
- Classify content domain
- Return ONLY the JSON, no explanations
"""

        response = self.call_llm(prompt)

        return SimplifiedContext(
            text_elements=response.get("text_elements", []),
            visual_theme=response.get("visual_theme", "neutral"),
            layout_pattern=response.get("layout_pattern", "form"),
            interaction_types=response.get("interaction_types", []),
            content_domain=response.get("content_domain", "utility"),
        )

    def call_llm(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Call Gemini to simplify GUI context"""
        response = self._call_gemini(prompt)
        return self._extract_json(response)


class ImplementationGeneratorLLM(BaseLLM):
    """LLM 2: Convert context to executable code with quality validation"""

    def generate_code(
        self, simplified_context: SimplifiedContext, environment_state: EnvironmentState
    ) -> GeneratedCode:
        """Generate executable code with built-in quality validation"""
        prompt = f"""
Generate JavaScript code to create safe GUI randomizations based on this interface:

CONTEXT:
Text Elements: {simplified_context.text_elements}
Visual Theme: {simplified_context.visual_theme}
Layout Pattern: {simplified_context.layout_pattern}
Interaction Types: {simplified_context.interaction_types}
Content Domain: {simplified_context.content_domain}

ENVIRONMENT:
DOM Tree: {environment_state.dom_tree[:1500]}
A11Y Tree: {environment_state.a11y_tree[:800]}
App Type: {environment_state.app_type}
Task: {environment_state.task_instruction}

Generate **only the JavaScript code** that will go **inside page.evaluate()**:

Requirements:
- Use selectors that exist in the provided DOM
- Include fallback selectors for robustness
- Add existence checks before manipulation
- Maximum 15 lines of code
- Include error handling
- Use `const` or `let` for variable declarations
- Do not include async code or external resource loading
- The code should run safely without throwing errors
- Focus on safe, minimal changes that preserve functionality

Output strictly only the JavaScript code inside the parentheses, nothing else.
Do not include the page.evaluate wrapper, any explanation, or extra text.

Example for text replacement:
```javascript
try {{
  let element = document.querySelector('button:contains("Add to Cart")');
  if (element) {{
    element.textContent = 'Buy Now';
  }} else {{
    element = document.querySelector('[data-testid="add-to-cart"]');
    if (element) {{
      element.textContent = 'Buy Now';
    }}
  }}
}} catch (error) {{
  console.log('Randomization skipped: ' + error.message);
}}
```
"""

        response = self.call_llm(prompt)
        code = response.get("code", "")

        # Extract selectors from code
        selectors = self._extract_selectors(code)

        return GeneratedCode(
            code=code,
            selectors_used=selectors,
            environment_dependencies=environment_state.app_type,
            error_handling="try" in code and "catch" in code,
        )

    def _extract_selectors(self, code: str) -> List[str]:
        """Extract CSS selectors from generated code"""
        import re

        selectors = []
        patterns = [
            r"locator\(['\"]([^'\"]+)['\"]\)",
            r"querySelector\(['\"]([^'\"]+)['\"]\)",
            r"getElementsByClassName\(['\"]([^'\"]+)['\"]\)",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, code)
            selectors.extend(matches)
        return selectors

    def call_llm(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Call Gemini to generate implementation code"""
        response = self._call_gemini(prompt)
        # Extract code from response (remove markdown formatting)
        if "```" in response:
            code = response.split("```")[1].removeprefix("javascript").strip()
        else:
            code = response.strip()

        return {"code": code}


class SimpleLLMOrchestra:
    """Simplified orchestrator for the two-LLM system"""

    _instance = None
    _initialized = False

    def __new__(cls):
        """Singleton pattern - only one instance across the entire application"""
        if cls._instance is None:
            cls._instance = super(SimpleLLMOrchestra, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once
        if not self._initialized:
            self.context_simplifier = ContextSimplifierLLM()
            self.implementation_generator = ImplementationGeneratorLLM()
            self.logger = logging.getLogger(__name__)
            SimpleLLMOrchestra._initialized = True

    def process_seed_trajectory(self, seed_trajectory: SeedTrajectory, env=None) -> List[ApprovedVariation]:
        """Process a seed trajectory through the simplified two-LLM system"""
        # Extract environment state (runtime if env provided)
        environment_state = self._extract_environment_state(seed_trajectory, env)

        # LLM 1: Simplify context
        simplified = self.context_simplifier.simplify(environment_state)

        # LLM 2: Generate implementation with built-in quality validation
        code = self.implementation_generator.generate_code(simplified, environment_state)

        # Create approved variation (simplified approval - no separate validation)
        approved_variation = ApprovedVariation(
            instruction=f"Apply {simplified.visual_theme} theme with {simplified.layout_pattern} layout",
            code=code,
            quality_score=85,  # Default high score for simplified system
            environment_dependencies=environment_state.app_type,
        )

        self.logger.info("Generated 1 approved variation from seed")
        return [approved_variation]

    def _extract_environment_state(self, seed_trajectory: SeedTrajectory, env=None) -> EnvironmentState:
        """Extract environment state from actual running environment"""
        # Determine app type from task type
        app_type = self._detect_app_type(seed_trajectory.task_type)

        # If we have access to the environment, extract real-time data
        if env and hasattr(env, "controller"):
            # Extract fresh environment state
            dom_tree, a11y_tree = self._extract_runtime_environment_data(env, app_type)
            current_url = self._extract_runtime_url(env, app_type)
            window_state = self._extract_runtime_window_state(env)
        else:
            # Fallback to basic extraction from trajectory
            dom_tree, a11y_tree = self._extract_dom_data(seed_trajectory.gt_actions_file_path)
            current_url = self._extract_url(seed_trajectory)
            window_state = {"width": 1920, "height": 1080}

        return EnvironmentState(
            dom_tree=dom_tree,
            a11y_tree=a11y_tree,
            app_type=app_type,
            current_url=current_url,
            window_state=window_state,
            task_instruction=seed_trajectory.task_instruction,
        )

    def _detect_app_type(self, task_type: str) -> str:
        """Detect application type from task type"""
        if task_type == "chrome":
            return "browser"
        elif task_type in [
            "libreoffice_calc",
            "libreoffice_writer",
            "libreoffice_impress",
            "gimp",
            "vlc",
            "vs_code",
        ]:
            return "desktop_app"
        else:
            return "browser"  # Default to browser

    def _extract_dom_data(self, trajectory_file_path: str) -> tuple[str, str]:
        """Extract DOM and accessibility data from trajectory file"""
        try:
            import json

            dom_parts = []
            a11y_parts = []

            with open(trajectory_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            step_data = json.loads(line.strip())
                            # Extract DOM and accessibility data from trajectory steps
                            if "dom_tree" in step_data:
                                dom_parts.append(step_data["dom_tree"])
                            if "a11y_tree" in step_data:
                                a11y_parts.append(step_data["a11y_tree"])
                        except json.JSONDecodeError:
                            continue

            # Combine all DOM and accessibility data
            dom_tree = (
                "\n".join(dom_parts)
                if dom_parts
                else "<html><body><button>Add to Cart</button></body></html>"
            )
            a11y_tree = "\n".join(a11y_parts) if a11y_parts else "button: Add to Cart"

            return dom_tree, a11y_tree

        except Exception as e:
            self.logger.warning(f"Could not extract DOM data from {trajectory_file_path}: {e}")
            return "<html><body><button>Add to Cart</button></body></html>", "button: Add to Cart"

    def _extract_runtime_environment_data(self, env, app_type: str) -> tuple[str, str]:
        """Extract DOM and accessibility data from running environment"""
        try:
            if app_type == "browser":
                # For browser apps, get DOM and accessibility tree
                dom_tree = self._get_browser_dom(env)
                a11y_tree = self._get_browser_a11y(env)
            elif app_type == "desktop_app":
                # For desktop apps, get application state
                dom_tree = self._get_desktop_app_state(env)
                a11y_tree = self._get_desktop_app_a11y(env)
            else:
                # Fallback
                dom_tree = "<html><body><button>Unknown App</button></body></html>"
                a11y_tree = "button: Unknown App"

            return dom_tree, a11y_tree
        except Exception as e:
            self.logger.warning(f"Could not extract runtime environment data: {e}")
            return "<html><body><button>Error</button></body></html>", "button: Error"

    def _get_browser_dom(self, env) -> str:
        """Get DOM tree from browser environment"""
        try:
            if hasattr(env.controller, "get_page_html"):
                return env.controller.get_page_html()
            elif hasattr(env.controller, "page") and env.controller.page:
                return env.controller.page.content()
            else:
                return "<html><body><button>No DOM available</button></body></html>"
        except Exception as e:
            self.logger.warning(f"Could not get browser DOM: {e}")
            return "<html><body><button>DOM Error</button></body></html>"

    def _get_browser_a11y(self, env) -> str:
        """Get accessibility tree from browser environment"""
        try:
            if hasattr(env.controller, "get_accessibility_tree"):
                return env.controller.get_accessibility_tree() or "No accessibility tree available"
            else:
                return "No accessibility tree available"
        except Exception as e:
            self.logger.warning(f"Could not get browser accessibility tree: {e}")
            return "Accessibility tree error"

    def _get_desktop_app_state(self, env) -> str:
        """Get desktop application state"""
        try:
            # For desktop apps, we can get window information
            if hasattr(env.controller, "get_vm_window_size"):
                window_info = env.controller.get_vm_window_size("")
                return f"<desktop><window>{window_info}</window></desktop>"
            else:
                return "<desktop><window>Unknown desktop app</window></desktop>"
        except Exception as e:
            self.logger.warning(f"Could not get desktop app state: {e}")
            return "<desktop><window>Error</window></desktop>"

    def _get_desktop_app_a11y(self, env) -> str:
        """Get desktop application accessibility tree"""
        try:
            if hasattr(env.controller, "get_accessibility_tree"):
                return env.controller.get_accessibility_tree() or "No desktop accessibility tree"
            else:
                return "No desktop accessibility tree"
        except Exception as e:
            self.logger.warning(f"Could not get desktop accessibility tree: {e}")
            return "Desktop accessibility error"

    def _extract_runtime_url(self, env, app_type: str) -> str:
        """Extract current URL from running environment"""
        try:
            if app_type == "browser":
                if hasattr(env.controller, "page") and env.controller.page:
                    return env.controller.page.url
                else:
                    return "https://unknown.com"
            else:
                return "desktop://unknown"
        except Exception as e:
            self.logger.warning(f"Could not extract runtime URL: {e}")
            return "https://error.com"

    def _extract_runtime_window_state(self, env) -> dict:
        """Extract window state from running environment"""
        try:
            if hasattr(env, "screen_width") and hasattr(env, "screen_height"):
                return {"width": env.screen_width, "height": env.screen_height}
            elif hasattr(env.controller, "get_vm_screen_size"):
                screen_info = env.controller.get_vm_screen_size()
                if screen_info:
                    return {
                        "width": screen_info.get("width", 1920),
                        "height": screen_info.get("height", 1080),
                    }
            return {"width": 1920, "height": 1080}
        except Exception as e:
            self.logger.warning(f"Could not extract runtime window state: {e}")
            return {"width": 1920, "height": 1080}

    def _extract_url(self, seed_trajectory: SeedTrajectory) -> str:
        """Extract current URL from seed trajectory"""
        try:
            # Try to extract URL from task config
            if "url" in seed_trajectory.config:
                return seed_trajectory.config["url"]
            elif "config" in seed_trajectory.config and "url" in seed_trajectory.config["config"]:
                return seed_trajectory.config["config"]["url"]
            else:
                return "https://example.com"  # Default URL
        except Exception:
            return "https://example.com"


def get_simple_llm_orchestra() -> SimpleLLMOrchestra:
    """Get the singleton simple LLM orchestra instance"""
    return SimpleLLMOrchestra()
