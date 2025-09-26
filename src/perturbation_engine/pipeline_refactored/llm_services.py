"""
LLM Services: Clean interfaces for LLM interactions
Following single responsibility principle
"""

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from google import genai
from google.genai import types

from perturbation_engine.pipeline_refactored.data_models import (
    CurriculumConfig,
    ExecutionContext,
    GeneratedTrajectory,
    PerturbationType,
    ScenarioSpec,
    SeedTrajectory,
)


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

    def extract_json(self, response: str) -> Dict[str, Any]:
        """
        Extract JSON from LLM response with robust parsing.
        Handles various formats: code blocks, plain JSON, mixed content.
        """
        try:
            # Strategy 3: Find JSON-like structures using regex
            # Look for objects or arrays that span multiple lines
            patterns = [
                r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",  # Nested objects
                r"\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]",  # Nested arrays
                r"\{.*?\}",  # Simple objects (non-greedy)
                r"\[.*?\]",  # Simple arrays (non-greedy)
            ]

            for pattern in patterns:
                matches = re.findall(pattern, response, re.DOTALL)
                for match in matches:
                    try:
                        result = json.loads(match.strip())
                        # Return the first valid JSON found
                        if result:  # Don't return empty objects/arrays
                            return result
                    except json.JSONDecodeError:
                        continue

            # Strategy 4: Character-by-character parsing for complex cases
            json_chars = {"{", "["}
            for i, char in enumerate(response):
                if char in json_chars:
                    # Find matching closing bracket/brace
                    bracket_count = 0
                    closing_char = "}" if char == "{" else "]"

                    for j in range(i, len(response)):
                        if response[j] == char:
                            bracket_count += 1
                        elif response[j] == closing_char:
                            bracket_count -= 1
                            if bracket_count == 0:
                                try:
                                    return json.loads(response[i : j + 1])
                                except json.JSONDecodeError:
                                    break

            self.logger.error("No valid JSON found in LLM response")
            return {}

        except Exception as e:
            self.logger.error(f"Unexpected error during JSON extraction: {e}")
            return {}


class CurriculumLLM(BaseLLM):
    """Generate scenario specs from seed trajectory"""

    def generate_scenario_specs(
        self,
        seed_trajectory: SeedTrajectory,
        app_states: List[Dict[str, Any]],
        curriculum_config: CurriculumConfig,
    ) -> List[ScenarioSpec]:
        """Generate curriculum of scenario specs"""

        # Group scenarios by app type
        app_scenarios = {}
        total_scenarios = curriculum_config.scenario_count

        # Distribute scenarios across app types
        for app_state in app_states:
            app_type = app_state.get("app_type", "unknown")
            if app_type != "unknown":
                app_scenarios[app_type] = self._generate_app_specific_scenarios(
                    app_type, seed_trajectory, app_state, curriculum_config
                )

        # Combine all scenarios
        all_scenarios = []
        for _, scenarios in app_scenarios.items():
            all_scenarios.extend(scenarios)

        # If we don't have enough scenarios, generate generic ones
        if len(all_scenarios) < total_scenarios:
            remaining = total_scenarios - len(all_scenarios)
            generic_scenarios = self._generate_generic_scenarios(
                seed_trajectory, remaining, curriculum_config
            )
            all_scenarios.extend(generic_scenarios)

        return all_scenarios[:total_scenarios]

    def _generate_app_specific_scenarios(
        self,
        app_type: str,
        seed_trajectory: SeedTrajectory,
        app_state: Dict[str, Any],
        curriculum_config: CurriculumConfig,
    ) -> List[ScenarioSpec]:
        """Generate scenarios specific to an app type"""

        if app_type == "browser":
            return self._generate_browser_scenarios(seed_trajectory, app_state, curriculum_config)
        elif app_type == "libreoffice":
            return self._generate_libreoffice_scenarios(seed_trajectory, app_state, curriculum_config)
        elif app_type in ["gimp", "image_editor"]:
            return self._generate_image_editor_scenarios(seed_trajectory, app_state, curriculum_config)
        elif app_type in ["file_manager", "file_browser"]:
            return self._generate_file_manager_scenarios(seed_trajectory, app_state, curriculum_config)
        else:
            return self._generate_generic_scenarios(seed_trajectory, 2, curriculum_config)

    def _generate_browser_scenarios(
        self, seed_trajectory: SeedTrajectory, app_state: Dict[str, Any], curriculum_config: CurriculumConfig
    ) -> List[ScenarioSpec]:
        """Generate browser-specific scenarios using JavaScript"""

        prompt = f"""
        Generate browser perturbation scenarios for this GUI task:

        Task: {seed_trajectory.task_instruction}
        App State: {app_state}

        Generate 3 scenario specifications for browser manipulation using JavaScript:

        AVAILABLE EXECUTOR: execute_js_on_page(js_code: str)
        - Input: Raw JavaScript code (NO markdown, NO ```, NO language tags)
        - Use: Theme changes, UI injection, element modification, layout changes

        REQUIRED JSON FORMAT (exactly these fields):
        {{
            "target_app": "browser",
            "perturbation_trigger": "string describing when to trigger",
            "available_perturbation_actions": "string with JavaScript code examples",
            "learning_objectives": "string describing what agent should learn",
            "target_components": ["array", "of", "ui", "components"],
            "perturbation_types": ["array", "of", "perturbation", "types"]
        }}

        EXAMPLES:
        - Theme change: "document.body.style.backgroundColor = 'darkblue'; document.querySelector('button').style.color = 'white';"
        - UI injection: "const newDiv = document.createElement('div'); newDiv.innerHTML = 'New Element'; document.body.appendChild(newDiv);"
        - Layout change: "document.querySelector('.container').style.flexDirection = 'column';"

        Return JSON array with exactly 3 scenario objects.
        """

        scenarios_data = self.call_llm(prompt)
        return self._parse_scenarios(scenarios_data, "browser")

    def _generate_libreoffice_scenarios(
        self, seed_trajectory: SeedTrajectory, app_state: Dict[str, Any], curriculum_config: CurriculumConfig
    ) -> List[ScenarioSpec]:
        """Generate LibreOffice-specific scenarios using UNO commands"""

        prompt = f"""
        Generate LibreOffice perturbation scenarios for this GUI task:

        Task: {seed_trajectory.task_instruction}
        App State: {app_state}

        Generate 3 scenario specifications for LibreOffice manipulation using UNO commands:

        AVAILABLE EXECUTOR: execute_uno_command(uno_code: str, parameters: Dict)
        - Input: Raw UNO Python code (NO markdown, NO ```, NO language tags)
        - Use: Spreadsheet operations, document changes, cell manipulation

        REQUIRED JSON FORMAT (exactly these fields):
        {{
            "target_app": "libreoffice",
            "perturbation_trigger": "string describing when to trigger",
            "available_perturbation_actions": "string with UNO code examples",
            "learning_objectives": "string describing what agent should learn",
            "target_components": ["array", "of", "ui", "components"],
            "perturbation_types": ["array", "of", "perturbation", "types"]
        }}

        EXAMPLES:
        - Cell manipulation: "doc = desktop.getCurrentComponent(); sheet = doc.getSheets().getByIndex(0); cell = sheet.getCellByPosition(0, 0); cell.setString('Hello');"
        - Theme change: "doc = desktop.getCurrentComponent(); doc.getCurrentController().getViewSettings().setPropertyValue('ShowGrid', False);"
        - Layout change: "doc = desktop.getCurrentComponent(); sheet = doc.getSheets().getByIndex(0); sheet.getColumns().getByIndex(0).setPropertyValue('Width', 2000);"

        Return JSON array with exactly 3 scenario objects.
        """

        scenarios_data = self.call_llm(prompt)
        return self._parse_scenarios(scenarios_data, "libreoffice")

    def _generate_image_editor_scenarios(
        self, seed_trajectory: SeedTrajectory, app_state: Dict[str, Any], curriculum_config: CurriculumConfig
    ) -> List[ScenarioSpec]:
        """Generate image editor scenarios using bash commands"""

        prompt = f"""
        Generate image editor perturbation scenarios for this GUI task:

        Task: {seed_trajectory.task_instruction}
        App State: {app_state}

        Generate 2 scenario specifications for image editor manipulation using bash commands:

        AVAILABLE EXECUTORS:
        - execute_bash_command(command: str): Raw bash commands
        - manipulate_app_state(parameters: Dict): App management

        REQUIRED JSON FORMAT (exactly these fields):
        {{
            "target_app": "gimp",
            "perturbation_trigger": "string describing when to trigger",
            "available_perturbation_actions": "string with bash/state manipulation examples",
            "learning_objectives": "string describing what agent should learn",
            "target_components": ["array", "of", "ui", "components"],
            "perturbation_types": ["array", "of", "perturbation", "types"]
        }}

        EXAMPLES:
        - Window resize: "wmctrl -r 'GIMP' -e 0,0,0,800,600"
        - App switch: {{"operation": "switch_to_app", "target_app": "gimp"}}
        - File operations: "cp /path/to/image.jpg /tmp/backup.jpg"

        Return JSON array with exactly 2 scenario objects.
        """

        scenarios_data = self.call_llm(prompt)
        return self._parse_scenarios(scenarios_data, "gimp")

    def _generate_file_manager_scenarios(
        self, seed_trajectory: SeedTrajectory, app_state: Dict[str, Any], curriculum_config: CurriculumConfig
    ) -> List[ScenarioSpec]:
        """Generate file manager scenarios using bash commands"""

        prompt = f"""
        Generate file manager perturbation scenarios for this GUI task:

        Task: {seed_trajectory.task_instruction}
        App State: {app_state}

        Generate 2 scenario specifications for file manager manipulation using bash commands:

        AVAILABLE EXECUTORS:
        - execute_bash_command(command: str): Raw bash commands
        - execute_python_command(python_code: str): Python automation

        REQUIRED JSON FORMAT (exactly these fields):
        {{
            "target_app": "file_manager",
            "perturbation_trigger": "string describing when to trigger",
            "available_perturbation_actions": "string with bash/python examples",
            "learning_objectives": "string describing what agent should learn",
            "target_components": ["array", "of", "ui", "components"],
            "perturbation_types": ["array", "of", "perturbation", "types"]
        }}

        EXAMPLES:
        - File operations: "mkdir -p /tmp/test_dir && touch /tmp/test_dir/file.txt"
        - Python automation: "import os; os.makedirs('/tmp/python_dir', exist_ok=True)"
        - Window management: "wmctrl -r 'Files' -e 0,0,0,1000,700"

        Return JSON array with exactly 2 scenario objects.
        """

        scenarios_data = self.call_llm(prompt)
        return self._parse_scenarios(scenarios_data, "file_manager")

    def _generate_generic_scenarios(
        self, seed_trajectory: SeedTrajectory, count: int, curriculum_config: CurriculumConfig
    ) -> List[ScenarioSpec]:
        """Generate generic scenarios using Python commands"""

        prompt = f"""
        Generate generic perturbation scenarios for this GUI task:

        Task: {seed_trajectory.task_instruction}

        Generate {count} scenario specifications using Python automation:

        AVAILABLE EXECUTOR: execute_python_command(python_code: str)
        - Input: Raw Python code (NO markdown, NO ```, NO language tags)
        - Use: System automation, data processing, general manipulation

        REQUIRED JSON FORMAT (exactly these fields):
        {{
            "target_app": "system",
            "perturbation_trigger": "string describing when to trigger",
            "available_perturbation_actions": "string with Python code examples",
            "learning_objectives": "string describing what agent should learn",
            "target_components": ["array", "of", "ui", "components"],
            "perturbation_types": ["array", "of", "perturbation", "types"]
        }}

        EXAMPLES:
        - System automation: "import subprocess; subprocess.run(['notify-send', 'Perturbation Applied'])"
        - Data processing: "import json; data = {{'perturbation': 'applied'}}; print(json.dumps(data))"
        - Window management: "import subprocess; subprocess.run(['wmctrl', '-a', 'Terminal'])"

        Return JSON array with exactly {count} scenario objects.
        """

        scenarios_data = self.call_llm(prompt)
        return self._parse_scenarios(scenarios_data, "system")

    def _parse_scenarios(self, scenarios_data: List[Dict[str, Any]], default_app: str) -> List[ScenarioSpec]:
        """Parse and validate scenario data with consistent format"""

        scenario_specs = []
        for i, scenario_data in enumerate(scenarios_data):
            try:
                # Ensure all required fields exist with defaults
                scenario_data = self._ensure_required_fields(scenario_data, default_app)

                # Parse perturbation types
                perturbation_types = []
                for pt_str in scenario_data.get("perturbation_types", []):
                    try:
                        perturbation_types.append(PerturbationType(pt_str))
                    except ValueError:
                        self.logger.warning(f"Unknown perturbation type: {pt_str}")

                scenario_spec = ScenarioSpec(
                    scenario_id=f"scenario_{i + 1}",
                    target_app=scenario_data.get("target_app", default_app),
                    perturbation_trigger=scenario_data.get("perturbation_trigger", ""),
                    available_perturbation_actions=scenario_data.get("available_perturbation_actions", ""),
                    learning_objectives=scenario_data.get("learning_objectives", ""),
                    target_components=scenario_data.get("target_components", []),
                    perturbation_types=perturbation_types,
                )
                scenario_specs.append(scenario_spec)
            except (ValueError, KeyError) as e:
                self.logger.error(f"Invalid scenario data: {e}")
                continue

        return scenario_specs

    def _ensure_required_fields(self, scenario_data: Dict[str, Any], default_app: str) -> Dict[str, Any]:
        """Ensure all required fields exist with proper defaults"""

        # Required fields with defaults
        defaults = {
            "target_app": default_app,
            "perturbation_trigger": "when user interacts with the application",
            "available_perturbation_actions": "execute_python_command('print(\"Perturbation applied\")')",
            "learning_objectives": "robustness to UI changes",
            "target_components": ["buttons", "menus"],
            "perturbation_types": ["theme", "layout"],
        }

        # Apply defaults for missing fields
        for key, default_value in defaults.items():
            if key not in scenario_data or not scenario_data[key]:
                scenario_data[key] = default_value

        return scenario_data

    def call_llm(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Call LLM to generate scenario specs"""
        response = self._call_gemini(prompt)
        return self.extract_json(response)


class PerturbationLLM(BaseLLM):
    """Decide when/how to perturb during runtime"""

    def decide_perturbation(
        self, execution_context: ExecutionContext, scenario_spec: ScenarioSpec
    ) -> Dict[str, Any]:
        """Decide whether to apply perturbation at current step"""

        # Route to app-specific perturbation decision
        target_app = scenario_spec.target_app.lower()

        if target_app == "browser":
            return self._decide_browser_perturbation(execution_context, scenario_spec)
        elif target_app == "libreoffice":
            return self._decide_libreoffice_perturbation(execution_context, scenario_spec)
        elif target_app in ["gimp", "image_editor"]:
            return self._decide_image_editor_perturbation(execution_context, scenario_spec)
        elif target_app in ["file_manager", "file_browser"]:
            return self._decide_file_manager_perturbation(execution_context, scenario_spec)
        else:
            return self._decide_generic_perturbation(execution_context, scenario_spec)

    def _decide_browser_perturbation(
        self, execution_context: ExecutionContext, scenario_spec: ScenarioSpec
    ) -> Dict[str, Any]:
        """Decide browser-specific perturbations using JavaScript"""

        prompt = f"""
        Decide whether to apply a browser perturbation during GUI task execution.

        CURRENT STATE:
        - Step: {execution_context.step_idx}
        - Action: {execution_context.current_action}
        - Action History: {execution_context.action_history[-3:] if execution_context.action_history else []}
        - CoT Context: {execution_context.cot_context}
        - App States: {execution_context.app_states}
        - Task: {execution_context.task_instruction}

        SCENARIO SPEC:
        - Target App: {scenario_spec.target_app}
        - Trigger: {scenario_spec.perturbation_trigger}
        - Available Actions: {scenario_spec.available_perturbation_actions}
        - Learning Objectives: {scenario_spec.learning_objectives}
        - Target Components: {scenario_spec.target_components}
        - Perturbation Types: {[pt.value for pt in scenario_spec.perturbation_types]}

        AVAILABLE EXECUTOR: execute_js_on_page(js_code: str)
        - Input: Raw JavaScript code (NO markdown, NO ```, NO language tags)
        - Use: Theme changes, UI injection, element modification, layout changes

        DECISION CRITERIA:
        1. Does the current step match the perturbation trigger?
        2. Is the browser active or relevant to the current action?
        3. What specific JavaScript perturbation should be applied?

        REQUIRED JSON FORMAT (exactly these fields):
        {{
            "should_apply": true/false,
            "perturbation_type": "theme_change" | "ui_injection" | "layout_change" | "element_modification",
            "target_app": "browser",
            "reasoning": "Brief explanation of why/why not to apply",
            "generated_code": "RAW_JAVASCRIPT_CODE_WITHOUT_MARKDOWN",
            "api_call": "execute_js_on_page",
            "parameters": {{"target_app": "browser"}}
        }}

        EXAMPLES:
        - Theme change: "document.body.style.backgroundColor = 'darkblue'; document.querySelector('button').style.color = 'white';"
        - UI injection: "const newDiv = document.createElement('div'); newDiv.innerHTML = 'New Element'; document.body.appendChild(newDiv);"
        - Layout change: "document.querySelector('.container').style.flexDirection = 'column';"

        Return JSON object with exactly the required fields.
        """

        response = self.call_llm(prompt)
        return self._validate_perturbation_decision(response)

    def _decide_libreoffice_perturbation(
        self, execution_context: ExecutionContext, scenario_spec: ScenarioSpec
    ) -> Dict[str, Any]:
        """Decide LibreOffice-specific perturbations using UNO commands"""

        prompt = f"""
        Decide whether to apply a LibreOffice perturbation during GUI task execution.

        CURRENT STATE:
        - Step: {execution_context.step_idx}
        - Action: {execution_context.current_action}
        - Action History: {execution_context.action_history[-3:] if execution_context.action_history else []}
        - CoT Context: {execution_context.cot_context}
        - App States: {execution_context.app_states}
        - Task: {execution_context.task_instruction}

        SCENARIO SPEC:
        - Target App: {scenario_spec.target_app}
        - Trigger: {scenario_spec.perturbation_trigger}
        - Available Actions: {scenario_spec.available_perturbation_actions}
        - Learning Objectives: {scenario_spec.learning_objectives}
        - Target Components: {scenario_spec.target_components}
        - Perturbation Types: {[pt.value for pt in scenario_spec.perturbation_types]}

        AVAILABLE EXECUTOR: execute_uno_command(uno_code: str, parameters: Dict)
        - Input: Raw UNO Python code (NO markdown, NO ```, NO language tags)
        - Use: Spreadsheet operations, document changes, cell manipulation

        DECISION CRITERIA:
        1. Does the current step match the perturbation trigger?
        2. Is LibreOffice active or relevant to the current action?
        3. What specific UNO perturbation should be applied?

        REQUIRED JSON FORMAT (exactly these fields):
        {{
            "should_apply": true/false,
            "perturbation_type": "cell_manipulation" | "theme_change" | "layout_change" | "content_variation",
            "target_app": "libreoffice",
            "reasoning": "Brief explanation of why/why not to apply",
            "generated_code": "RAW_UNO_PYTHON_CODE_WITHOUT_MARKDOWN",
            "api_call": "execute_uno_command",
            "parameters": {{"target_app": "libreoffice"}}
        }}

        EXAMPLES:
        - Cell manipulation: "doc = desktop.getCurrentComponent(); sheet = doc.getSheets().getByIndex(0); cell = sheet.getCellByPosition(0, 0); cell.setString('Hello');"
        - Theme change: "doc = desktop.getCurrentComponent(); doc.getCurrentController().getViewSettings().setPropertyValue('ShowGrid', False);"
        - Layout change: "doc = desktop.getCurrentComponent(); sheet = doc.getSheets().getByIndex(0); sheet.getColumns().getByIndex(0).setPropertyValue('Width', 2000);"

        Return JSON object with exactly the required fields.
        """

        response = self.call_llm(prompt)
        return self._validate_perturbation_decision(response)

    def _decide_image_editor_perturbation(
        self, execution_context: ExecutionContext, scenario_spec: ScenarioSpec
    ) -> Dict[str, Any]:
        """Decide image editor perturbations using bash commands and app state manipulation"""

        prompt = f"""
        Decide whether to apply an image editor perturbation during GUI task execution.

        CURRENT STATE:
        - Step: {execution_context.step_idx}
        - Action: {execution_context.current_action}
        - Action History: {execution_context.action_history[-3:] if execution_context.action_history else []}
        - CoT Context: {execution_context.cot_context}
        - App States: {execution_context.app_states}
        - Task: {execution_context.task_instruction}

        SCENARIO SPEC:
        - Target App: {scenario_spec.target_app}
        - Trigger: {scenario_spec.perturbation_trigger}
        - Available Actions: {scenario_spec.available_perturbation_actions}
        - Learning Objectives: {scenario_spec.learning_objectives}
        - Target Components: {scenario_spec.target_components}
        - Perturbation Types: {[pt.value for pt in scenario_spec.perturbation_types]}

        AVAILABLE EXECUTORS:
        - execute_bash_command(command: str): Raw bash commands
        - manipulate_app_state(parameters: Dict): App management

        DECISION CRITERIA:
        1. Does the current step match the perturbation trigger?
        2. Is the image editor active or relevant to the current action?
        3. What specific perturbation should be applied?

        REQUIRED JSON FORMAT (exactly these fields):
        {{
            "should_apply": true/false,
            "perturbation_type": "window_resize" | "app_switch" | "file_operation" | "layout_change",
            "target_app": "gimp",
            "reasoning": "Brief explanation of why/why not to apply",
            "generated_code": "RAW_BASH_COMMAND_OR_EMPTY_FOR_APP_STATE",
            "api_call": "execute_bash_command" | "manipulate_app_state",
            "parameters": {{"target_app": "gimp", "operation": "resize_window", "width": 800, "height": 600}}
        }}

        EXAMPLES:
        - Window resize: "wmctrl -r 'GIMP' -e 0,0,0,800,600"
        - App switch: Use manipulate_app_state with {{"operation": "switch_to_app", "target_app": "gimp"}}
        - File operations: "cp /path/to/image.jpg /tmp/backup.jpg"

        Return JSON object with exactly the required fields.
        """

        response = self.call_llm(prompt)
        return self._validate_perturbation_decision(response)

    def _decide_file_manager_perturbation(
        self, execution_context: ExecutionContext, scenario_spec: ScenarioSpec
    ) -> Dict[str, Any]:
        """Decide file manager perturbations using bash and Python commands"""

        prompt = f"""
        Decide whether to apply a file manager perturbation during GUI task execution.

        CURRENT STATE:
        - Step: {execution_context.step_idx}
        - Action: {execution_context.current_action}
        - Action History: {execution_context.action_history[-3:] if execution_context.action_history else []}
        - CoT Context: {execution_context.cot_context}
        - App States: {execution_context.app_states}
        - Task: {execution_context.task_instruction}

        SCENARIO SPEC:
        - Target App: {scenario_spec.target_app}
        - Trigger: {scenario_spec.perturbation_trigger}
        - Available Actions: {scenario_spec.available_perturbation_actions}
        - Learning Objectives: {scenario_spec.learning_objectives}
        - Target Components: {scenario_spec.target_components}
        - Perturbation Types: {[pt.value for pt in scenario_spec.perturbation_types]}

        AVAILABLE EXECUTORS:
        - execute_bash_command(command: str): Raw bash commands
        - execute_python_command(python_code: str): Python automation

        DECISION CRITERIA:
        1. Does the current step match the perturbation trigger?
        2. Is the file manager active or relevant to the current action?
        3. What specific perturbation should be applied?

        REQUIRED JSON FORMAT (exactly these fields):
        {{
            "should_apply": true/false,
            "perturbation_type": "file_operation" | "window_management" | "automation" | "layout_change",
            "target_app": "file_manager",
            "reasoning": "Brief explanation of why/why not to apply",
            "generated_code": "RAW_BASH_OR_PYTHON_CODE_WITHOUT_MARKDOWN",
            "api_call": "execute_bash_command" | "execute_python_command",
            "parameters": {{"target_app": "file_manager"}}
        }}

        EXAMPLES:
        - File operations: "mkdir -p /tmp/test_dir && touch /tmp/test_dir/file.txt"
        - Python automation: "import os; os.makedirs('/tmp/python_dir', exist_ok=True)"
        - Window management: "wmctrl -r 'Files' -e 0,0,0,1000,700"

        Return JSON object with exactly the required fields.
        """

        response = self.call_llm(prompt)
        return self._validate_perturbation_decision(response)

    def _decide_generic_perturbation(
        self, execution_context: ExecutionContext, scenario_spec: ScenarioSpec
    ) -> Dict[str, Any]:
        """Decide generic perturbations using Python commands"""

        prompt = f"""
        Decide whether to apply a generic perturbation during GUI task execution.

        CURRENT STATE:
        - Step: {execution_context.step_idx}
        - Action: {execution_context.current_action}
        - Action History: {execution_context.action_history[-3:] if execution_context.action_history else []}
        - CoT Context: {execution_context.cot_context}
        - App States: {execution_context.app_states}
        - Task: {execution_context.task_instruction}

        SCENARIO SPEC:
        - Target App: {scenario_spec.target_app}
        - Trigger: {scenario_spec.perturbation_trigger}
        - Available Actions: {scenario_spec.available_perturbation_actions}
        - Learning Objectives: {scenario_spec.learning_objectives}
        - Target Components: {scenario_spec.target_components}
        - Perturbation Types: {[pt.value for pt in scenario_spec.perturbation_types]}

        AVAILABLE EXECUTOR: execute_python_command(python_code: str)
        - Input: Raw Python code (NO markdown, NO ```, NO language tags)
        - Use: System automation, data processing, general manipulation

        DECISION CRITERIA:
        1. Does the current step match the perturbation trigger?
        2. Is the target app active or relevant to the current action?
        3. What specific Python perturbation should be applied?

        REQUIRED JSON FORMAT (exactly these fields):
        {{
            "should_apply": true/false,
            "perturbation_type": "system_automation" | "data_processing" | "window_management" | "general_manipulation",
            "target_app": "{scenario_spec.target_app}",
            "reasoning": "Brief explanation of why/why not to apply",
            "generated_code": "RAW_PYTHON_CODE_WITHOUT_MARKDOWN",
            "api_call": "execute_python_command",
            "parameters": {{"target_app": "{scenario_spec.target_app}"}}
        }}

        EXAMPLES:
        - System automation: "import subprocess; subprocess.run(['notify-send', 'Perturbation Applied'])"
        - Data processing: "import json; data = {{'perturbation': 'applied'}}; print(json.dumps(data))"
        - Window management: "import subprocess; subprocess.run(['wmctrl', '-a', 'Terminal'])"

        Return JSON object with exactly the required fields.
        """

        response = self.call_llm(prompt)
        return self._validate_perturbation_decision(response)

    def call_llm(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Call LLM to make perturbation decision"""
        response = self._call_gemini(prompt)
        result = self.extract_json(response)

        # Validate and clean the response
        if isinstance(result, dict):
            result = self._validate_perturbation_decision(result)

        return result

    def _validate_perturbation_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean perturbation decision format"""
        try:
            # Ensure required fields exist
            if "should_apply" not in decision:
                decision["should_apply"] = False

            if "api_call" not in decision:
                decision["api_call"] = "execute_python_command"

            # Clean generated_code if it exists
            if "generated_code" in decision and decision["generated_code"]:
                code = decision["generated_code"]
                # Remove markdown formatting
                if "```" in code:
                    parts = code.split("```")
                    if len(parts) > 1:
                        # Take the code between ``` markers
                        code = parts[1].strip()
                        # Remove language tags
                        if code.startswith(("javascript", "python", "bash", "js")):
                            code = code.split("\n", 1)[1] if "\n" in code else ""
                decision["generated_code"] = code.strip()

            # Validate api_call
            valid_api_calls = [
                "execute_js_on_page",
                "execute_bash_command",
                "execute_python_command",
                "execute_uno_command",
                "manipulate_app_state",
            ]
            if decision["api_call"] not in valid_api_calls:
                decision["api_call"] = "execute_python_command"

            # Ensure parameters exist for manipulate_app_state
            if decision["api_call"] == "manipulate_app_state":
                if "parameters" not in decision:
                    decision["parameters"] = {}
                if "operation" not in decision["parameters"]:
                    decision["parameters"]["operation"] = "switch_to_app"
                if "target_app" not in decision["parameters"]:
                    decision["parameters"]["target_app"] = decision.get("target_app", "unknown")

            return decision

        except Exception as e:
            self.logger.error(f"Error validating perturbation decision: {e}")
            return {
                "should_apply": False,
                "perturbation_type": "unknown",
                "target_app": "unknown",
                "reasoning": "Validation error",
                "generated_code": "",
                "api_call": "execute_python_command",
                "parameters": {},
            }


class QualityEvaluationLLM(BaseLLM):
    """Score trajectory quality"""

    def evaluate_trajectory_quality(
        self, generated_trajectory: GeneratedTrajectory, scenario_spec: ScenarioSpec
    ) -> float:
        """Evaluate the quality of a generated trajectory"""

        prompt = f"""
        Evaluate the quality of this generated trajectory:

        TRAJECTORY:
        - Success: {generated_trajectory.success}
        - Generation Time: {generated_trajectory.generation_time}
        - Perturbations Applied: {len(generated_trajectory.perturbation_log)}

        SCENARIO SPEC:
        - Target App: {scenario_spec.target_app}
        - Learning Objectives: {scenario_spec.learning_objectives}
        - Target Components: {scenario_spec.target_components}
        - Perturbation Types: {[pt.value for pt in scenario_spec.perturbation_types]}
        - Available Actions: {scenario_spec.available_perturbation_actions}

        PERTURBATION LOG:
        {json.dumps(generated_trajectory.perturbation_log, indent=2)}

        EVALUATION CRITERIA:
        1. Task completion success (did the agent complete the original task?)
        2. Learning objective achievement (did perturbations help achieve learning goals?)
        3. Perturbation effectiveness (were the applied perturbations appropriate?)
        4. Robustness demonstration (did the agent adapt to changes?)
        5. Code execution success (did the generated code work correctly?)

        Rate the trajectory quality on a scale of 0.0 to 1.0 based on:
        - 0.0-0.3: Poor (task failed, no learning, ineffective perturbations)
        - 0.3-0.6: Fair (partial success, some learning, basic perturbations)
        - 0.6-0.8: Good (task completed, clear learning, effective perturbations)
        - 0.8-1.0: Excellent (robust completion, strong learning, sophisticated perturbations)

        Return JSON:
        {{
            "quality_score": 0.0-1.0,
            "reasoning": "Detailed explanation of score based on criteria",
            "strengths": ["specific strength 1", "specific strength 2"],
            "weaknesses": ["specific weakness 1", "specific weakness 2"],
            "learning_achievement": "How well learning objectives were met",
            "perturbation_effectiveness": "How effective the perturbations were"
        }}
        """

        response = self.call_llm(prompt)
        return response.get("quality_score", 0.0)

    def call_llm(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Call LLM to evaluate trajectory quality"""
        response = self._call_gemini(prompt)
        return self.extract_json(response)
