"""
LLM Services: Clean interfaces for LLM interactions
Following single responsibility principle
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from google import genai
from google.genai import types

from perturbation_engine.pipeline.data_models import (
    CurriculumConfig,
    ExecutionContext,
    GeneratedTrajectory,
    PerturbationType,
    ScenarioSpec,
    SeedTrajectory,
)


class BaseLLM(ABC):
    """Base class for all LLM components"""

    def __init__(self, model_name: str = "gemini-2.0-flash-lite"):
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

    def extract_json(self, response: str) -> List[Dict[str, Any]]:
        """
        Extract JSON from LLM response with robust parsing.
        Handles various formats: code blocks, plain JSON, mixed content.
        """
        self.logger.debug(f"extract_json called with response length: {len(response)}")
        self.logger.debug(f"extract_json response preview: {response[:300]}...")

        try:
            # Use character-by-character parsing to find complete JSON structures
            # This avoids the overlapping regex pattern issue
            results = []
            json_chars = {"{", "["}
            processed_positions = set()  # Track processed positions to avoid duplicates

            i = 0
            while i < len(response):
                char = response[i]
                if char in json_chars and i not in processed_positions:
                    # Find matching closing bracket/brace
                    bracket_count = 0
                    closing_char = "}" if char == "{" else "]"
                    start_pos = i

                    for j in range(i, len(response)):
                        if response[j] == char:
                            bracket_count += 1
                        elif response[j] == closing_char:
                            bracket_count -= 1
                            if bracket_count == 0:
                                # Found complete JSON structure
                                try:
                                    json_str = response[start_pos : j + 1].strip()
                                    parsed = json.loads(json_str)
                                    self.logger.debug(
                                        f"Found complete JSON at position {start_pos}-{j}: {type(parsed)}"
                                    )

                                    # Add to results if it's valid
                                    if isinstance(parsed, list) and len(parsed) > 0:
                                        self.logger.debug(f"Found array with {len(parsed)} items")
                                        results.append(parsed)
                                        # Mark all positions in this range as processed
                                        for pos in range(start_pos, j + 1):
                                            processed_positions.add(pos)
                                        # If we found an array, we're done (prioritize arrays)
                                        break
                                    elif isinstance(parsed, dict) and parsed:
                                        self.logger.debug("Found object")
                                        results.append(parsed)
                                        # Mark all positions in this range as processed
                                        for pos in range(start_pos, j + 1):
                                            processed_positions.add(pos)

                                except json.JSONDecodeError as e:
                                    self.logger.debug(f"JSON decode error at position {start_pos}-{j}: {e}")

                                # Move past this JSON structure
                                i = j + 1
                                break
                    else:
                        # No matching closing bracket found, move to next character
                        i += 1
                else:
                    i += 1
            if len(results) == 0:
                self.logger.error("No valid JSON found in LLM response")
                return []

            self.logger.debug(f"extract_json returning {len(results)} results")
            self.logger.debug(f"extract_json result types: {[type(r) for r in results]}")

            # Prioritize arrays over individual objects
            # If we found an array, use only that (it should contain all the scenario objects)
            array_results = [r for r in results if isinstance(r, list)]
            if array_results:
                self.logger.debug(f"Found {len(array_results)} arrays, using the first one")
                primary_array = array_results[0]
                self.logger.debug(f"Primary array has {len(primary_array)} items")

                # Extract individual dictionaries from the array
                flattened_results = []
                for item in primary_array:
                    if isinstance(item, dict):
                        flattened_results.append(item)
                    else:
                        self.logger.debug(f"Skipping non-dict item in array: {type(item)}")

                self.logger.debug(f"extract_json final flattened results: {len(flattened_results)} items")
                return flattened_results
            else:
                # No array found, use individual objects
                self.logger.debug("No arrays found, using individual objects")
                object_results = [r for r in results if isinstance(r, dict)]
                self.logger.debug(f"extract_json final object results: {len(object_results)} items")
                return object_results
        except Exception as e:
            self.logger.error(f"Unexpected error during JSON extraction: {e}")
            return []


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
        # save inputs to file for faster debugging iterations
        # debug_inputs = []
        # for app_state in app_states:
        #     app_type = app_state.get("app_type", "unknown")
        #     if app_type != "unknown":
        #         seed_trajectory_dict = {
        #             "task_id": seed_trajectory.task_id,
        #             "task_type": seed_trajectory.task_type,
        #             "task_instruction": seed_trajectory.task_instruction,
        #             "config": seed_trajectory.config,
        #             "gt_actions_file_path": seed_trajectory.gt_actions_file_path,
        #             "gt_actions": seed_trajectory.gt_actions
        #         }
        #         curriculum_config_dict = {
        #             "scenario_count": curriculum_config.scenario_count,
        #             "num_parallel_vms": curriculum_config.num_parallel_vms,
        #             "result_base_dir": curriculum_config.result_base_dir,
        #             "beginner_scenarios": curriculum_config.beginner_scenarios,
        #             "intermediate_scenarios": curriculum_config.intermediate_scenarios,
        #             "advanced_scenarios": curriculum_config.advanced_scenarios
        #         }
        #         debug_inputs.append({
        #             "app_type": app_type,
        #             "seed_trajectory": seed_trajectory_dict,
        #             "app_state": app_state,
        #             "curriculum_config": curriculum_config_dict
        #         })

        # with open("inputs.json", "w") as f:
        #     json.dump(debug_inputs, f, indent=2)

        for app_state in app_states:
            app_type = app_state.get("app_type", "unknown")
            self.logger.debug(f"Processing app_state with app_type: {app_type}")
            self.logger.debug(f"app_state: {app_state}")
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

        self.logger.debug(f"_generate_app_specific_scenarios called with app_type: {app_type}")
        self.logger.debug(f"app_state: {app_state}")

        if app_type == "browser":
            return self._generate_browser_scenarios(seed_trajectory, app_state, curriculum_config)
        elif app_type == "libreoffice":
            return self._generate_libreoffice_scenarios(seed_trajectory, app_state, curriculum_config)
        elif app_type in ["gimp", "image_editor"]:
            return self._generate_image_editor_scenarios(seed_trajectory, app_state, curriculum_config)
        elif app_type in ["file_manager", "file_browser"]:
            return self._generate_file_manager_scenarios(seed_trajectory, app_state, curriculum_config)
        elif app_type in ["terminal"]:
            return self._generate_terminal_scenarios(seed_trajectory, app_state, curriculum_config)
        elif app_type in ["vs_code"]:
            return self._generate_vs_code_scenarios(seed_trajectory, app_state, curriculum_config)
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
        - Use: Background theme changes, non-intrusive UI modifications
        - API Call: execute_js_on_page

        IMPORTANT: Focus on BACKGROUND browser environment manipulation that won't interfere with the main task:
        - Background color changes, non-critical UI styling
        - Adding background elements that don't block main content
        - Subtle theme modifications that don't affect functionality
        - Background animations or effects that don't interfere with user interactions

        REQUIRED JSON FORMAT (exactly these fields):
        {{
            "target_app": "browser",
            "perturbation_trigger": "string describing when to trigger",
            "available_perturbation_actions": "string with JavaScript code examples for background manipulation",
            "learning_objectives": "string describing what agent should learn about robustness",
            "target_components": ["array", "of", "background", "components"],
            "perturbation_types": ["theme", "layout", "ui_injection"]
        }}

        VALID PERTURBATION TYPES: theme, layout, content_variation, ui_injection, notification, background_process, window_management, file_operations

        EXAMPLES (background manipulation only):
        - Background theme: "document.body.style.backgroundColor = '#f0f0f0'; document.body.style.transition = 'background-color 0.3s';"
        - Background decoration: "const bg = document.createElement('div'); bg.style.position = 'fixed'; bg.style.top = '0'; bg.style.right = '0'; bg.style.width = '50px'; bg.style.height = '50px'; bg.style.backgroundColor = 'rgba(0,0,0,0.1)'; document.body.appendChild(bg);"
        - Background animation: "document.body.style.animation = 'subtle-pulse 2s infinite';"

        Return JSON array with a list of exactly 3 scenario objects.
        """

        scenarios_data = self.call_llm(prompt)
        return self._parse_scenarios(scenarios_data, "browser")

    def _generate_libreoffice_scenarios(
        self, seed_trajectory: SeedTrajectory, app_state: Dict[str, Any], curriculum_config: CurriculumConfig
    ) -> List[ScenarioSpec]:
        """Generate LibreOffice-specific scenarios using UNO commands"""

        self.logger.debug("_generate_libreoffice_scenarios called")
        self.logger.debug(f"seed_trajectory.task_instruction: {seed_trajectory.task_instruction}")
        self.logger.debug(f"app_state: {app_state}")

        prompt = f"""
        Generate LibreOffice perturbation scenarios for this GUI task:

        Task: {seed_trajectory.task_instruction}
        App State: {app_state}

        Generate 3 scenario specifications for LibreOffice manipulation using UNO commands:

        AVAILABLE EXECUTOR: execute_uno_command(uno_code: str, parameters: Dict)
        - Input: Raw UNO Python code (NO markdown, NO ```, NO language tags)
        - Use: Background document styling, non-critical UI modifications
        - API Call: execute_uno_command

        IMPORTANT: Focus on BACKGROUND LibreOffice environment manipulation that won't interfere with the main task:
        - Background document styling, non-critical formatting changes
        - UI theme modifications that don't affect functionality
        - Background grid/display settings that don't impact data
        - Subtle visual changes that don't interfere with user workflow

        REQUIRED JSON FORMAT (exactly these fields):
        {{
            "target_app": "libreoffice",
            "perturbation_trigger": "string describing when to trigger",
            "available_perturbation_actions": "string with UNO code examples for background manipulation",
            "learning_objectives": "string describing what agent should learn about robustness",
            "target_components": ["array", "of", "background", "components"],
            "perturbation_types": ["theme", "layout", "ui_injection"]
        }}

        VALID PERTURBATION TYPES: theme, layout, content_variation, ui_injection, notification, background_process, window_management, file_operations

        EXAMPLES (background manipulation only):
        - Background theme: "doc = desktop.getCurrentComponent(); viewSettings = doc.getCurrentController().getViewSettings(); viewSettings.setPropertyValue('ShowGrid', False);"
        - Background styling: "doc = desktop.getCurrentComponent(); sheet = doc.getSheets().getByIndex(0); column = sheet.getColumns().getByIndex(0); column.setPropertyValue('Width', 2000);"
        - UI theme: "doc = desktop.getCurrentComponent(); viewSettings = doc.getCurrentController().getViewSettings(); viewSettings.setPropertyValue('ShowFormulaBar', False);"
        - Background color: "doc = desktop.getCurrentComponent(); sheet = doc.getSheets().getByIndex(0); cell = sheet.getCellByPosition(0, 0); cell.CellBackColor = 0xFFFF00;"
        - Grid settings: "doc = desktop.getCurrentComponent(); viewSettings = doc.getCurrentController().getViewSettings(); viewSettings.setPropertyValue('ShowGrid', True); viewSettings.setPropertyValue('GridVisible', True);"
        - Cell formatting: "doc = desktop.getCurrentComponent(); sheet = doc.getSheets().getByIndex(0); cell = sheet.getCellByPosition(1, 1); cell.String = 'Background Data'; cell.CharWeight = 150;"
        - Sheet properties: "doc = desktop.getCurrentComponent(); sheet = doc.getSheets().getByIndex(0); sheet.setPropertyValue('IsVisible', True);"

        Return JSON array with a list of exactly 3 scenario objects.
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
        - API Calls: execute_bash_command, manipulate_app_state

        IMPORTANT: Focus on BACKGROUND desktop environment manipulation that won't interfere with the main task:
        - Background file operations, temporary file creation
        - Window management of OTHER applications (not the main task)
        - System notifications, background processes
        - Desktop environment modifications that don't affect the primary workflow

        REQUIRED JSON FORMAT (exactly these fields):
        {{
            "target_app": "gimp",
            "perturbation_trigger": "string describing when to trigger",
            "available_perturbation_actions": "string with bash/state manipulation examples for background manipulation",
            "learning_objectives": "string describing what agent should learn about robustness",
            "target_components": ["array", "of", "background", "components"],
            "perturbation_types": ["notification", "background_process", "file_operations"]
        }}

        VALID PERTURBATION TYPES: theme, layout, content_variation, ui_injection, notification, background_process, window_management, file_operations

        EXAMPLES (background manipulation only):
        - Background files: "mkdir -p /tmp/background_images && touch /tmp/background_images/temp.jpg"
        - Other window management: "wmctrl -r 'Calculator' -e 0,100,100,300,200"
        - System notifications: "notify-send 'Background Process' 'Image processing complete'"
        - Background processes: "nohup sleep 30 > /dev/null 2>&1 &"
        - Desktop wallpaper: "gsettings set org.gnome.desktop.background picture-uri 'file:///usr/share/backgrounds/gnome/adwaita-morning.jpg'"

        Return JSON array with a list of exactly 2 scenario objects.
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
        - API Calls: execute_bash_command, execute_python_command

        IMPORTANT: Focus on BACKGROUND desktop environment manipulation that won't interfere with the main task:
        - Background file operations, temporary directory creation
        - System notifications, background processes
        - Window management of OTHER applications (not the main task)
        - Desktop environment modifications that don't affect the primary workflow

        REQUIRED JSON FORMAT (exactly these fields):
        {{
            "target_app": "file_manager",
            "perturbation_trigger": "string describing when to trigger",
            "available_perturbation_actions": "string with bash/python examples for background manipulation",
            "learning_objectives": "string describing what agent should learn about robustness",
            "target_components": ["array", "of", "background", "components"],
            "perturbation_types": ["notification", "background_process", "file_operations"]
        }}

        VALID PERTURBATION TYPES: theme, layout, content_variation, ui_injection, notification, background_process, window_management, file_operations

        EXAMPLES (background manipulation only):
        - Background files: "mkdir -p /tmp/background_work && touch /tmp/background_work/process.log"
        - System notifications: "notify-send 'Background Process' 'File sync complete'"
        - Other window management: "wmctrl -r 'Calculator' -e 0,100,100,300,200"
        - Background processes: "nohup find /tmp -name '*.tmp' -delete > /dev/null 2>&1 &"
        - Desktop settings: "gsettings set org.gnome.desktop.interface clock-format '24h'"

        Return JSON array with a list of exactly 2 scenario objects.
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
        - Use: Background system automation, non-intrusive modifications
        - API Call: execute_python_command

        IMPORTANT: Focus on BACKGROUND desktop environment manipulation that won't interfere with the main task:
        - System notifications, background processes
        - Desktop theme changes, background settings
        - Window management of OTHER applications (not the main task)
        - Background file operations, temporary data creation

        REQUIRED JSON FORMAT (exactly these fields):
        {{
            "target_app": "system",
            "perturbation_trigger": "string describing when to trigger",
            "available_perturbation_actions": "string with Python code examples for background manipulation",
            "learning_objectives": "string describing what agent should learn about robustness",
            "target_components": ["array", "of", "background", "components"],
            "perturbation_types": ["notification", "background_process", "window_management"]
        }}

        VALID PERTURBATION TYPES: theme, layout, content_variation, ui_injection, notification, background_process, window_management, file_operations

        EXAMPLES (background manipulation only):
        - System notifications: "import subprocess; subprocess.run(['notify-send', 'Background Process', 'System update running'])"
        - Background files: "import os; os.makedirs('/tmp/background_work', exist_ok=True); open('/tmp/background_work/process.log', 'w').write('Background process started')"
        - Other window management: "import subprocess; subprocess.run(['wmctrl', '-r', 'Calculator', '-e', '0,100,100,300,200'])"
        - Background processes: "import subprocess; subprocess.Popen(['sleep', '60'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)"
        - Desktop settings: "import subprocess; subprocess.run(['gsettings', 'set', 'org.gnome.desktop.interface', 'gtk-theme', 'Adwaita-dark'])"

        Return JSON array with a list of exactly {count} scenario objects.
        """

        scenarios_data = self.call_llm(prompt)
        return self._parse_scenarios(scenarios_data, "system")

    def _generate_terminal_scenarios(
        self, seed_trajectory: SeedTrajectory, app_state: Dict[str, Any], curriculum_config: CurriculumConfig
    ) -> List[ScenarioSpec]:
        """Generate terminal-specific scenarios using bash commands"""

        prompt = f"""
        Generate terminal perturbation scenarios for this GUI task:

        Task: {seed_trajectory.task_instruction}
        App State: {app_state}

        Generate 2 scenario specifications for terminal manipulation using bash commands:

        AVAILABLE EXECUTORS:
        - execute_bash_command(command: str): Raw bash commands
        - execute_python_command(python_code: str): Python automation
        - API Calls: execute_bash_command, execute_python_command

        IMPORTANT: Focus on BACKGROUND desktop environment manipulation that won't interfere with the main task:
        - System notifications, background processes, desktop themes
        - Window management of OTHER applications (not the main task)
        - Background file operations, system settings changes
        - Desktop environment modifications that don't affect the primary workflow

        REQUIRED JSON FORMAT (exactly these fields):
        {{
            "target_app": "terminal",
            "perturbation_trigger": "string describing when to trigger",
            "available_perturbation_actions": "string with bash/python examples for background manipulation",
            "learning_objectives": "string describing what agent should learn about robustness",
            "target_components": ["array", "of", "background", "components"],
            "perturbation_types": ["notification", "background_process", "window_management"]
        }}

        VALID PERTURBATION TYPES: theme, layout, content_variation, ui_injection, notification, background_process, window_management, file_operations

        EXAMPLES (background manipulation only):
        - System notifications: "notify-send 'Background Process' 'System update running'"
        - Desktop theme: "gsettings set org.gnome.desktop.interface gtk-theme 'Adwaita-dark'"
        - Background files: "mkdir -p /tmp/background_work && touch /tmp/background_work/process.log"
        - Other window management: "wmctrl -r 'Calculator' -e 0,100,100,300,200"
        - Background processes: "nohup sleep 30 > /dev/null 2>&1 &"
        - Desktop settings: "gsettings set org.gnome.desktop.interface clock-format '24h'"

        Return JSON array with a list of exactly 2 scenario objects.
        """

        scenarios_data = self.call_llm(prompt)
        return self._parse_scenarios(scenarios_data, "terminal")

    def _generate_vs_code_scenarios(
        self, seed_trajectory: SeedTrajectory, app_state: Dict[str, Any], curriculum_config: CurriculumConfig
    ) -> List[ScenarioSpec]:
        """Generate VS Code-specific scenarios using Python automation"""

        prompt = f"""
        Generate VS Code perturbation scenarios for this GUI task:

        Task: {seed_trajectory.task_instruction}
        App State: {app_state}

        Generate 2 scenario specifications for VS Code manipulation using Python automation:

        AVAILABLE EXECUTORS:
        - execute_python_command(python_code: str): Python automation
        - execute_bash_command(command: str): Raw bash commands
        - API Calls: execute_python_command, execute_bash_command

        IMPORTANT: Focus on BACKGROUND VS Code environment manipulation that won't interfere with the main task:
        - Background file operations, temporary file creation
        - Window management, panel resizing, sidebar toggling
        - System notifications, background processes
        - Desktop environment modifications that don't affect the primary workflow

        REQUIRED JSON FORMAT (exactly these fields):
        {{
            "target_app": "vs_code",
            "perturbation_trigger": "string describing when to trigger",
            "available_perturbation_actions": "string with python/bash examples for background manipulation",
            "learning_objectives": "string describing what agent should learn about robustness",
            "target_components": ["array", "of", "background", "components"],
            "perturbation_types": ["notification", "background_process", "window_management"]
        }}

        VALID PERTURBATION TYPES: theme, layout, content_variation, ui_injection, notification, background_process, window_management, file_operations

        EXAMPLES (background manipulation only):
        - Background files: "import os; os.makedirs('/tmp/vscode_temp', exist_ok=True); open('/tmp/vscode_temp/debug.log', 'w').write('Background process started')"
        - Window management: "wmctrl -r 'Visual Studio Code' -e 0,0,0,1200,800"
        - Settings: "import json; settings = {{'workbench.colorTheme': 'Dark+'}}; print(json.dumps(settings))"
        - System notifications: "import subprocess; subprocess.run(['notify-send', 'VS Code', 'Background process completed'])"
        - Background processes: "import subprocess; subprocess.Popen(['sleep', '10'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)"
        - Desktop settings: "import subprocess; subprocess.run(['gsettings', 'set', 'org.gnome.desktop.interface', 'gtk-theme', 'Adwaita-dark'])"
        - Background cleanup: "import os; [os.remove(os.path.join('/tmp', f)) for f in os.listdir('/tmp') if f.endswith('.tmp')]"

        Return JSON array with a list of exactly 2 scenario objects.
        """

        scenarios_data = self.call_llm(prompt)
        return self._parse_scenarios(scenarios_data, "vs_code")

    def _parse_scenarios(self, scenarios_data: List[Dict[str, Any]], default_app: str) -> List[ScenarioSpec]:
        """Parse and validate scenario data with consistent format"""

        self.logger.debug(f"_parse_scenarios called with scenarios_data type: {type(scenarios_data)}")
        self.logger.debug(
            f"scenarios_data length: {len(scenarios_data) if isinstance(scenarios_data, list) else 'N/A'}"
        )
        self.logger.debug(f"default_app: {default_app}")

        scenario_specs = []
        for i, scenario_data in enumerate(scenarios_data):
            self.logger.debug(f"Processing scenario {i}, type: {type(scenario_data)}")
            self.logger.debug(f"scenario_data content: {scenario_data}")

            # Defensive check: if scenario_data is still a list, skip it
            if isinstance(scenario_data, list):
                self.logger.warning(f"Skipping scenario {i} because it's still a list after flattening")
                continue

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

        # Check if scenario_data is unexpectedly a list
        if isinstance(scenario_data, list):
            self.logger.warning("_ensure_required_fields received LIST instead of DICT!")
            self.logger.warning(f"scenario_data type: {type(scenario_data)}")
            self.logger.warning(
                f"scenario_data length: {len(scenario_data) if isinstance(scenario_data, list) else 'N/A'}"
            )
            self.logger.warning(f"scenario_data content: {scenario_data}")
            self.logger.warning(f"default_app: {default_app}")

            # Try to handle the list case by taking the first element
            if len(scenario_data) > 0:
                self.logger.warning(f"Taking first element from list: {scenario_data[0]}")
                scenario_data = scenario_data[0]
            else:
                self.logger.warning("Empty list, creating default dict")
                scenario_data = {}
        elif not isinstance(scenario_data, dict):
            self.logger.warning(f"_ensure_required_fields received unexpected type: {type(scenario_data)}")
            self.logger.warning(f"scenario_data content: {scenario_data}")
            scenario_data = {}

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

    def call_llm(self, prompt: str, **kwargs) -> List[Dict[str, Any]]:
        """Call LLM to generate scenario specs"""
        response = self._call_gemini(prompt)
        self.logger.debug(f"call_llm raw response: {response[:200]}...")
        result = self.extract_json(response)
        self.logger.debug(f"call_llm extract_json result type: {type(result)}")
        self.logger.debug(
            f"call_llm extract_json result length: {len(result) if isinstance(result, list) else 'N/A'}"
        )
        self.logger.debug(f"call_llm extract_json result content: {result}")
        return result


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
        elif target_app == "terminal":
            return self._decide_terminal_perturbation(execution_context, scenario_spec)
        elif target_app == "vs_code":
            return self._decide_vs_code_perturbation(execution_context, scenario_spec)
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

    def _decide_terminal_perturbation(
        self, execution_context: ExecutionContext, scenario_spec: ScenarioSpec
    ) -> Dict[str, Any]:
        """Decide terminal perturbations using bash and Python commands"""

        prompt = f"""
        Decide whether to apply a terminal perturbation during GUI task execution.

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

        IMPORTANT: Focus on BACKGROUND desktop environment manipulation that won't interfere with the main task:
        - System notifications, background processes, desktop themes
        - Window management of OTHER applications (not the main task)
        - Background file operations, system settings changes
        - Desktop environment modifications that don't affect the primary workflow

        DECISION CRITERIA:
        1. Does the current step match the perturbation trigger?
        2. Is the terminal active or relevant to the current action?
        3. What specific background perturbation should be applied?

        REQUIRED JSON FORMAT (exactly these fields):
        {{
            "should_apply": true/false,
            "perturbation_type": "system_notification" | "background_process" | "desktop_theme" | "other_window_management",
            "target_app": "terminal",
            "reasoning": "Brief explanation of why/why not to apply",
            "generated_code": "RAW_BASH_OR_PYTHON_CODE_WITHOUT_MARKDOWN",
            "api_call": "execute_bash_command" | "execute_python_command",
            "parameters": {{"target_app": "terminal"}}
        }}

        EXAMPLES (background manipulation only):
        - System notifications: "notify-send 'Background Process' 'System update running'"
        - Desktop theme: "gsettings set org.gnome.desktop.interface gtk-theme 'Adwaita-dark'"
        - Background files: "mkdir -p /tmp/background_work && touch /tmp/background_work/process.log"
        - Other window management: "wmctrl -r 'Calculator' -e 0,100,100,300,200"

        Return JSON object with exactly the required fields.
        """

        response = self.call_llm(prompt)
        return self._validate_perturbation_decision(response)

    def _decide_vs_code_perturbation(
        self, execution_context: ExecutionContext, scenario_spec: ScenarioSpec
    ) -> Dict[str, Any]:
        """Decide VS Code perturbations using Python automation and bash commands"""

        prompt = f"""
        Decide whether to apply a VS Code perturbation during GUI task execution.

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
        - execute_python_command(python_code: str): Python automation
        - execute_bash_command(command: str): Raw bash commands

        IMPORTANT: Focus on BACKGROUND VS Code environment manipulation that won't interfere with the main task:
        - Theme changes, extension settings, workspace configurations
        - Background file operations, temporary file creation
        - Window management, panel resizing, sidebar toggling
        - Settings modifications that don't affect the primary workflow

        DECISION CRITERIA:
        1. Does the current step match the perturbation trigger?
        2. Is VS Code active or relevant to the current action?
        3. What specific background perturbation should be applied?

        REQUIRED JSON FORMAT (exactly these fields):
        {{
            "should_apply": true/false,
            "perturbation_type": "theme_change" | "background_files" | "window_management" | "settings_modification",
            "target_app": "vs_code",
            "reasoning": "Brief explanation of why/why not to apply",
            "generated_code": "RAW_PYTHON_OR_BASH_CODE_WITHOUT_MARKDOWN",
            "api_call": "execute_python_command" | "execute_bash_command",
            "parameters": {{"target_app": "vs_code"}}
        }}

        EXAMPLES (background manipulation only):
        - Theme change: "import subprocess; subprocess.run(['code', '--install-extension', 'theme-extension'])"
        - Background files: "import os; os.makedirs('/tmp/vscode_temp', exist_ok=True)"
        - Window management: "wmctrl -r 'Visual Studio Code' -e 0,0,0,1200,800"
        - Settings: "import json; settings = {{'workbench.colorTheme': 'Dark+'}}; print(json.dumps(settings))"

        Return JSON object with exactly the required fields.
        """

        response = self.call_llm(prompt)
        return self._validate_perturbation_decision(response)

    def call_llm(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Call LLM to make perturbation decision"""
        response = self._call_gemini(prompt)
        result = self.extract_json(response)

        # Validate and clean the response
        if isinstance(result, list) and len(result) > 0:
            result = result[0]  # Take the first result if it's a list
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
        result = self.extract_json(response)

        # Handle case where extract_json returns a list
        if isinstance(result, list) and len(result) > 0:
            return result[0]  # Take the first result if it's a list
        return result


if __name__ == "__main__":
    llm = CurriculumLLM()

    with open("inputs.json", "r") as f:
        inputs = json.load(f)

    app_states = []

    for input in inputs:
        app_type = input["app_type"]
        seed_trajectory = input["seed_trajectory"]
        seed_trajectory = SeedTrajectory(**seed_trajectory)
        app_state = input["app_state"]
        curriculum_config = input["curriculum_config"]
        curriculum_config = CurriculumConfig(**curriculum_config)

        app_states.append(app_state)

    llm.generate_scenario_specs(seed_trajectory, app_states, curriculum_config)
