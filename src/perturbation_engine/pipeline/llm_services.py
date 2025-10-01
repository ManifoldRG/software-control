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

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        # Ensure logging is configured for subprocess (only if not already configured)
        if not logging.getLogger().handlers:
            from perturbation_engine.configure_logging import configure_logging

            configure_logging()

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
        retries = 0
        while retries < 3:
            if not self.client:
                return self._get_mock_response()

            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_budget=0)
                    ),
                )
                return response.text
            except Exception as e:
                self.logger.error(f"Error calling Gemini: {e}")
                retries += 1

        self.logger.error("Failed to call Gemini after 3 retries")
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
                                    # Clean up common JSON formatting issues
                                    json_str = json_str.replace("\n", " ").replace("\r", " ")
                                    # Remove extra whitespace
                                    json_str = " ".join(json_str.split())
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
                                    self.logger.debug(f"Problematic JSON string: {json_str[:200]}...")
                                    # Try to fix common issues
                                    try:
                                        # Remove any trailing commas
                                        json_str = json_str.replace(",}", "}").replace(",]", "]")
                                        # Try parsing again
                                        parsed = json.loads(json_str)
                                        self.logger.debug("Successfully parsed after fixing trailing commas")
                                        # Continue with the parsed result
                                        if isinstance(parsed, list) and len(parsed) > 0:
                                            self.logger.debug(f"Found array with {len(parsed)} items")
                                            results.append(parsed)
                                            for pos in range(start_pos, j + 1):
                                                processed_positions.add(pos)
                                            break
                                        elif isinstance(parsed, dict) and parsed:
                                            self.logger.debug("Found object")
                                            results.append(parsed)
                                            for pos in range(start_pos, j + 1):
                                                processed_positions.add(pos)
                                    except json.JSONDecodeError as e2:
                                        self.logger.debug(f"Still failed after fixing: {e2}")

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

    def _build_common_prompt_sections(
        self, app_name: str, executor_info: Dict[str, str], scenario_types: List[str], examples: List[str]
    ) -> str:
        """Build efficient prompt sections for scenario spec generation"""

        core_requirements = """
        ═══════════════════════════════════════════════════════════════
        CORE REQUIREMENTS: Visual-Only Perturbations
        ═══════════════════════════════════════════════════════════════

        ✅ DO: Modify colors, fonts, layouts, themes, spacing, borders
        ✅ DO: Use professional design systems (Material, Fluent, HIG, Ant, HighContrast)
        ✅ DO: Target 15+ elements, affect 3+ visual dimensions
        ✅ DO: Generate ORIGINAL code (not copied from examples)

        ❌ DON'T: Interfere with task functionality
        ❌ DON'T: Touch target forms, buttons, or navigation
        ❌ DON'T: Break the main task workflow
        ❌ DON'T: Copy examples - create novel approaches
        """

        design_systems = """
        🎨 DESIGN SYSTEMS TO USE:
        Material 3: #6750A4, #625B71, Roboto, 4-20px radius
        Fluent: #0078D4, Segoe UI, 0-8px radius
        Apple HIG: #007AFF, SF Pro, 4-16px radius
        Ant Design: #1890FF, Roboto, 2-8px radius
        High Contrast: #000/#FFF, Arial, 0px radius, 2-4px borders

        Pick ONE system per scenario, apply consistently across all elements.
        """

        visual_dimensions = """
        📏 TARGET 3+ DIMENSIONS:
        1. COLOR: Backgrounds, text, borders, accents
        2. TYPOGRAPHY: Fonts, sizes (12-24px), weights (300-900)
        3. LAYOUT: Padding (4-48px), margins, alignment
        4. SHAPE: Radius (0-24px), shadows, borders
        5. MOTION: Transitions (100-500ms) - subtle only
        6. DEPTH: Z-index, overlays, elevation
        7. DENSITY: Compact vs Spacious modes
        8. SEMANTICS: Primary/secondary hierarchy
        """

        json_format = f"""
        REQUIRED JSON FORMAT:
        {{
            "target_app": "{app_name}",
            "perturbation_trigger": "when to apply (e.g., 'at task start', 'during interaction')",
            "available_perturbation_actions": "COMPLETE DESIGN SYSTEM {executor_info["language"]} code targeting specific elements",
            "learning_objectives": "visual invariance learning goal",
            "target_components": ["specific", "UI", "components"],
            "perturbation_types": ["theme", "layout", "content_variation"]
        }}

        VALID TYPES: theme, layout, content_variation, ui_injection, notification, background_process
        """

        scenario_types_section = f"SCENARIO TYPES FOR {app_name.upper()}:\n"
        for i, scenario_type in enumerate(scenario_types, 1):
            scenario_types_section += f"        {i}. {scenario_type}\n"

        examples_section = f"EXAMPLES ({app_name}) - INSPIRATION ONLY:\n"
        for example in examples:
            examples_section += f"        {example}\n"

        return f"""
        {core_requirements}

        {design_systems}

        {visual_dimensions}

        {scenario_types_section}

        {json_format}

        {examples_section}
        """

    def _get_app_specific_config(self, app_name: str) -> Dict[str, Any]:
        """Get app-specific configuration for prompt building"""
        configs = {
            "browser": {
                "executor": {
                    "name": "execute_js_on_page",
                    "language": "JavaScript",
                    "description": "Complete design system transformations for visual invariance learning",
                },
                "scenario_types": [
                    "Complete Design System: Apply Material/Fluent/HIG themes to all interactive elements",
                    "Density Variations: Compact/comfortable/spacious modes affecting padding and spacing",
                    "Typography Systems: Change font families, sizes, weights across headings and content",
                    "Color Palette Transformations: Systematic color changes using professional palettes",
                ],
                "examples": [
                    "Material Design 3: const md3={primary:'#6750A4',surface:'#FFFBFE'}; document.querySelectorAll('button').forEach((b,i)=>{ b.style.backgroundColor=i%2?md3.primary:'#625B71'; b.style.color='#FFF'; b.style.borderRadius='20px'; b.style.padding='10px 24px'; b.style.fontFamily='Roboto'; });",
                    "Fluent Design: const fluent={primary:'#0078D4',bg:'#F3F2F1'}; document.body.style.backgroundColor=fluent.bg; document.querySelectorAll('button, input').forEach(el=>{ el.style.backgroundColor=fluent.primary; el.style.fontFamily='Segoe UI'; el.style.borderRadius='2px'; });",
                    "High Contrast: const hc={bg:'#000',fg:'#FFF',link:'#FF0'}; document.body.style.backgroundColor=hc.bg; document.body.style.color=hc.fg; document.querySelectorAll('a').forEach(a=>{ a.style.color=hc.link; a.style.fontWeight='700'; });",
                ],
            },
            "libreoffice_calc": {
                "executor": {
                    "name": "execute_uno_command",
                    "language": "UNO Python",
                    "description": "Complete spreadsheet theme transformations for visual invariance learning",
                },
                "scenario_types": [
                    "Professional Themes: Finance/Marketing/Engineering color schemes for grid styling",
                    "Grid Appearance: Modify grid colors, borders, view settings systematically",
                    "Typography Variations: Change cell fonts, sizes, weights across rows/columns",
                    "Density Modes: Compact/comfortable zoom levels and spacing",
                ],
                "examples": [
                    "Finance Theme: doc=desktop.getCurrentComponent(); sheet=doc.getSheets().getByIndex(0); [sheet.getCellByPosition(c,0).CellBackColor=0x0F4C81 for c in range(10)]; viewSettings=doc.getCurrentController().getViewSettings(); viewSettings.setPropertyValue('GridColor',0xCDD5DE)",
                    "Marketing Theme: doc=desktop.getCurrentComponent(); sheet=doc.getSheets().getByIndex(0); [sheet.getCellByPosition(c,0).CellBackColor=0xFF6B35 for c in range(8)]; viewSettings.setPropertyValue('ZoomValue',120)",
                    "Dark Theme: doc=desktop.getCurrentComponent(); sheet=doc.getSheets().getByIndex(0); [[sheet.getCellByPosition(c,r).CellBackColor=0x2D2D2D for c in range(12)] for r in range(20)]; viewSettings.setPropertyValue('GridColor',0x3C3C3C)",
                ],
            },
            "libreoffice_impress": {
                "executor": {
                    "name": "execute_uno_command",
                    "language": "UNO Python",
                    "description": "Complete presentation theme transformations for visual invariance learning",
                },
                "scenario_types": [
                    "Presentation Themes: Corporate/Academic/Creative background schemes",
                    "Slide Appearance: Modify slide colors, view modes systematically",
                    "View Configurations: Change zoom, panes, navigation appearance",
                    "Design Consistency: Apply cohesive themes across multiple slides",
                ],
                "examples": [
                    "Corporate Theme: doc=desktop.getCurrentComponent(); slides=doc.getDrawPages(); [slides.getByIndex(i).setPropertyValue('BackgroundColor',0xF8F9FA) for i in range(min(slides.getCount(),10))]; viewSettings=doc.getCurrentController().getViewSettings(); viewSettings.setPropertyValue('ZoomType',1)",
                    "Academic Theme: doc=desktop.getCurrentComponent(); slides=doc.getDrawPages(); [slides.getByIndex(i).setPropertyValue('BackgroundColor',0xFFFBF5) for i in range(min(slides.getCount(),5))]; viewSettings.setPropertyValue('ShowNotesPane',True)",
                    "Creative Theme: colors=[0xFFE5E5,0xE5F5FF,0xFFF5E5,0xF0E5FF]; [doc.getDrawPages().getByIndex(i).setPropertyValue('BackgroundColor',colors[i]) for i in range(min(len(colors),doc.getDrawPages().getCount()))]",
                ],
            },
            "libreoffice_writer": {
                "executor": {
                    "name": "execute_uno_command",
                    "language": "UNO Python",
                    "description": "Complete document theme transformations for visual invariance learning",
                },
                "scenario_types": [
                    "Document Themes: Professional/Academic/Creative page styling",
                    "Page Layouts: Modify margins, headers, footers systematically",
                    "View Configurations: Change zoom, ruler, status bar appearance",
                    "Background Variations: Page colors and visual settings",
                ],
                "examples": [
                    "Professional Theme: doc=desktop.getCurrentComponent(); pageStyles=doc.getStyleFamilies().getByName('PageStyles'); standardPage=pageStyles.getByName('Standard'); standardPage.setPropertyValue('BackColor',0xFFFFFF); standardPage.setPropertyValue('HeaderIsOn',True); viewSettings=doc.getCurrentController().getViewSettings(); viewSettings.setPropertyValue('ShowRuler',True)",
                    "Academic Theme: standardPage.setPropertyValue('BackColor',0xFFFBF5); standardPage.setPropertyValue('LeftMargin',3000); viewSettings.setPropertyValue('ShowTextBoundaries',True); viewSettings.setPropertyValue('ZoomValue',120)",
                    "Creative Theme: standardPage.setPropertyValue('BackColor',0xFFF8E1); standardPage.setPropertyValue('LeftMargin',2000); viewSettings.setPropertyValue('ShowRuler',False); viewSettings.setPropertyValue('ZoomType',3)",
                ],
            },
            "gimp": {
                "executor": {
                    "name": "execute_system_perturbation",
                    "language": "system",
                    "description": "System theme transformations for background environment diversity",
                },
                "scenario_types": [
                    "Desktop Theme Complete: Light/Dark/HighContrast system-wide changes",
                    "Background Notifications: System update/file sync notifications",
                ],
                "examples": [
                    "Theme Combo: gsettings set org.gnome.desktop.interface gtk-theme 'Adwaita-dark'; gsettings set org.gnome.desktop.interface icon-theme 'Papirus-Dark'; gsettings set org.gnome.desktop.interface color-scheme 'prefer-dark'",
                ],
            },
            "file_manager": {
                "executor": {
                    "name": "execute_system_perturbation",
                    "language": "system",
                    "description": "System theme and window management for background diversity",
                },
                "scenario_types": [
                    "Theme + Window: System theme changes with window positioning",
                    "Background Files: Temporary file creation in background",
                ],
                "examples": [
                    "Combined: gsettings set org.gnome.desktop.interface gtk-theme 'Adwaita-dark'; wmctrl -r 'Files' -e 0,100,100,1000,700; notify-send 'System' 'Environment updated'",
                ],
            },
            "terminal": {
                "executor": {
                    "name": "execute_system_perturbation",
                    "language": "system",
                    "description": "Background system changes that don't affect terminal tasks",
                },
                "scenario_types": [
                    "Background System: Theme changes and notifications only",
                    "Desktop Environment: System-level modifications",
                ],
                "examples": [
                    "Background: gsettings set org.gnome.desktop.interface gtk-theme 'Adwaita-dark'; notify-send 'Background' 'System task running'; mkdir -p /tmp/bg_proc",
                ],
            },
            "vs_code": {
                "executor": {
                    "name": "execute_system_perturbation",
                    "language": "system",
                    "description": "Background system changes that don't affect VS Code tasks",
                },
                "scenario_types": [
                    "Background Operations: File and system changes only",
                    "Desktop Theming: System-level theme modifications",
                ],
                "examples": [
                    "Background: gsettings set org.gnome.desktop.interface gtk-theme 'Adwaita-dark'; mkdir -p /tmp/vscode_bg; notify-send 'Background' 'Process running'",
                ],
            },
            "system": {
                "executor": {
                    "name": "execute_system_perturbation",
                    "language": "system",
                    "description": "Background desktop environment modifications",
                },
                "scenario_types": [
                    "System Theme: Complete desktop theme transformations",
                    "Background Processes: Notifications and background operations",
                ],
                "examples": [
                    "Complete Theme: gsettings set org.gnome.desktop.interface gtk-theme 'Adwaita-dark'; gsettings set org.gnome.desktop.interface icon-theme 'Papirus-Dark'; notify-send 'System' 'Theme updated'",
                ],
            },
        }
        return configs.get(app_name, configs["browser"])  # Default to browser config

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
        elif app_type == "libreoffice_calc":
            return self._generate_libreoffice_calc_scenarios(seed_trajectory, app_state, curriculum_config)
        elif app_type == "libreoffice_impress":
            return self._generate_libreoffice_impress_scenarios(seed_trajectory, app_state, curriculum_config)
        elif app_type == "libreoffice_writer":
            return self._generate_libreoffice_writer_scenarios(seed_trajectory, app_state, curriculum_config)
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

        config = self._get_app_specific_config("browser")
        common_sections = self._build_common_prompt_sections(
            "browser", config["executor"], config["scenario_types"], config["examples"]
        )

        prompt = f"""
        Generate sophisticated browser perturbation scenarios for this GUI task:

        Task: {seed_trajectory.task_instruction}
        App State: {app_state}

        Generate 3 scenario specifications for browser manipulation using JavaScript:

        AVAILABLE EXECUTOR: {config["executor"]["name"]}({config["executor"]["language"].lower()}_code: str)
        - Input: Raw {config["executor"]["language"]} code (NO markdown, NO ```, NO language tags)
        - Use: {config["executor"]["description"]}
        - API Call: {config["executor"]["name"]}

        {common_sections}

        Return JSON array with a list of exactly 3 scenario objects.
        """

        scenarios_data = self.call_llm(prompt)
        return self._parse_scenarios(scenarios_data, "browser")

    def _generate_libreoffice_calc_scenarios(
        self, seed_trajectory: SeedTrajectory, app_state: Dict[str, Any], curriculum_config: CurriculumConfig
    ) -> List[ScenarioSpec]:
        """Generate LibreOffice Calc-specific scenarios using UNO commands"""

        self.logger.debug("_generate_libreoffice_calc_scenarios called")
        self.logger.debug(f"seed_trajectory.task_instruction: {seed_trajectory.task_instruction}")
        self.logger.debug(f"app_state: {app_state}")

        config = self._get_app_specific_config("libreoffice_calc")
        common_sections = self._build_common_prompt_sections(
            "libreoffice_calc", config["executor"], config["scenario_types"], config["examples"]
        )

        prompt = f"""
        Generate sophisticated LibreOffice Calc perturbation scenarios for this GUI task:

        Task: {seed_trajectory.task_instruction}
        App State: {app_state}

        Generate 3 scenario specifications for LibreOffice Calc manipulation using UNO commands:

        AVAILABLE EXECUTOR: {config["executor"]["name"]}({config["executor"]["language"].lower()}_code: str, parameters: Dict)
        - Input: Raw {config["executor"]["language"]} code (NO markdown, NO ```, NO language tags)
        - Use: {config["executor"]["description"]}
        - API Call: {config["executor"]["name"]}

        {common_sections}

        Return JSON array with a list of exactly 3 scenario objects.
        """

        scenarios_data = self.call_llm(prompt)
        return self._parse_scenarios(scenarios_data, "libreoffice_calc")

    def _generate_libreoffice_impress_scenarios(
        self, seed_trajectory: SeedTrajectory, app_state: Dict[str, Any], curriculum_config: CurriculumConfig
    ) -> List[ScenarioSpec]:
        """Generate LibreOffice Impress-specific scenarios using UNO commands"""

        self.logger.debug("_generate_libreoffice_impress_scenarios called")
        self.logger.debug(f"seed_trajectory.task_instruction: {seed_trajectory.task_instruction}")
        self.logger.debug(f"app_state: {app_state}")

        config = self._get_app_specific_config("libreoffice_impress")
        common_sections = self._build_common_prompt_sections(
            "libreoffice_impress", config["executor"], config["scenario_types"], config["examples"]
        )

        prompt = f"""
        Generate sophisticated LibreOffice Impress perturbation scenarios for this GUI task:

        Task: {seed_trajectory.task_instruction}
        App State: {app_state}

        Generate 3 scenario specifications for LibreOffice Impress manipulation using UNO commands:

        AVAILABLE EXECUTOR: {config["executor"]["name"]}({config["executor"]["language"].lower()}_code: str, parameters: Dict)
        - Input: Raw {config["executor"]["language"]} code (NO markdown, NO ```, NO language tags)
        - Use: {config["executor"]["description"]}
        - API Call: {config["executor"]["name"]}

        {common_sections}

        Return JSON array with a list of exactly 3 scenario objects.
        """

        scenarios_data = self.call_llm(prompt)
        return self._parse_scenarios(scenarios_data, "libreoffice_impress")

    def _generate_libreoffice_writer_scenarios(
        self, seed_trajectory: SeedTrajectory, app_state: Dict[str, Any], curriculum_config: CurriculumConfig
    ) -> List[ScenarioSpec]:
        """Generate LibreOffice Writer-specific scenarios using UNO commands"""

        self.logger.debug("_generate_libreoffice_writer_scenarios called")
        self.logger.debug(f"seed_trajectory.task_instruction: {seed_trajectory.task_instruction}")
        self.logger.debug(f"app_state: {app_state}")

        config = self._get_app_specific_config("libreoffice_writer")
        common_sections = self._build_common_prompt_sections(
            "libreoffice_writer", config["executor"], config["scenario_types"], config["examples"]
        )

        prompt = f"""
        Generate sophisticated LibreOffice Writer perturbation scenarios for this GUI task:

        Task: {seed_trajectory.task_instruction}
        App State: {app_state}

        Generate 3 scenario specifications for LibreOffice Writer manipulation using UNO commands:

        AVAILABLE EXECUTOR: {config["executor"]["name"]}({config["executor"]["language"].lower()}_code: str, parameters: Dict)
        - Input: Raw {config["executor"]["language"]} code (NO markdown, NO ```, NO language tags)
        - Use: {config["executor"]["description"]}
        - API Call: {config["executor"]["name"]}

        {common_sections}

        Return JSON array with a list of exactly 3 scenario objects.
        """

        scenarios_data = self.call_llm(prompt)
        return self._parse_scenarios(scenarios_data, "libreoffice_writer")

    def _generate_image_editor_scenarios(
        self, seed_trajectory: SeedTrajectory, app_state: Dict[str, Any], curriculum_config: CurriculumConfig
    ) -> List[ScenarioSpec]:
        """Generate image editor scenarios using bash commands"""

        config = self._get_app_specific_config("gimp")
        common_sections = self._build_common_prompt_sections(
            "gimp", config["executor"], config["scenario_types"], config["examples"]
        )

        prompt = f"""
        Generate sophisticated image editor perturbation scenarios for this GUI task:

        Task: {seed_trajectory.task_instruction}
        App State: {app_state}

        Generate 2 scenario specifications for image editor manipulation using bash commands:

        AVAILABLE EXECUTORS:
        - {config["executor"]["name"]}(command: str): Raw {config["executor"]["language"]} commands
        - manipulate_app_state(parameters: Dict): App management
        - API Calls: {config["executor"]["name"]}, manipulate_app_state

        {common_sections}

        Return JSON array with a list of exactly 2 scenario objects.
        """

        scenarios_data = self.call_llm(prompt)
        return self._parse_scenarios(scenarios_data, "gimp")

    def _generate_file_manager_scenarios(
        self, seed_trajectory: SeedTrajectory, app_state: Dict[str, Any], curriculum_config: CurriculumConfig
    ) -> List[ScenarioSpec]:
        """Generate file manager scenarios using bash commands"""

        config = self._get_app_specific_config("file_manager")
        common_sections = self._build_common_prompt_sections(
            "file_manager", config["executor"], config["scenario_types"], config["examples"]
        )

        prompt = f"""
        Generate sophisticated file manager perturbation scenarios for this GUI task:

        Task: {seed_trajectory.task_instruction}
        App State: {app_state}

        Generate 2 scenario specifications for file manager manipulation using bash commands:

        AVAILABLE EXECUTORS:
        - {config["executor"]["name"]}(command: str): Raw {config["executor"]["language"]} commands
        - execute_python_command(python_code: str): Python automation
        - API Calls: {config["executor"]["name"]}, execute_python_command

        {common_sections}

        Return JSON array with a list of exactly 2 scenario objects.
        """

        scenarios_data = self.call_llm(prompt)
        return self._parse_scenarios(scenarios_data, "file_manager")

    def _generate_generic_scenarios(
        self, seed_trajectory: SeedTrajectory, count: int, curriculum_config: CurriculumConfig
    ) -> List[ScenarioSpec]:
        """Generate generic scenarios using Python commands"""

        config = self._get_app_specific_config("system")
        common_sections = self._build_common_prompt_sections(
            "system", config["executor"], config["scenario_types"], config["examples"]
        )

        prompt = f"""
        Generate generic perturbation scenarios for this GUI task:

        Task: {seed_trajectory.task_instruction}

        Generate {count} scenario specifications using Python automation:

        AVAILABLE EXECUTOR: {config["executor"]["name"]}({config["executor"]["language"].lower()}_code: str)
        - Input: Raw {config["executor"]["language"]} code (NO markdown, NO ```, NO language tags)
        - Use: {config["executor"]["description"]}
        - API Call: {config["executor"]["name"]}

        IMPORTANT: Focus on BACKGROUND desktop environment manipulation that won't interfere with the main task:
        - System notifications, background processes
        - Desktop theme changes, background settings
        - Window management of OTHER applications (not the main task)
        - Background file operations, temporary data creation

        {common_sections}

        Return JSON array with a list of exactly {count} scenario objects.
        """

        scenarios_data = self.call_llm(prompt)
        return self._parse_scenarios(scenarios_data, "system")

    def _generate_terminal_scenarios(
        self, seed_trajectory: SeedTrajectory, app_state: Dict[str, Any], curriculum_config: CurriculumConfig
    ) -> List[ScenarioSpec]:
        """Generate terminal-specific scenarios using bash commands"""

        config = self._get_app_specific_config("terminal")
        common_sections = self._build_common_prompt_sections(
            "terminal", config["executor"], config["scenario_types"], config["examples"]
        )

        prompt = f"""
        Generate terminal perturbation scenarios for this GUI task:

        Task: {seed_trajectory.task_instruction}
        App State: {app_state}

        Generate 2 scenario specifications for terminal manipulation using bash commands:

        AVAILABLE EXECUTORS:
        - {config["executor"]["name"]}(command: str): Raw {config["executor"]["language"]} commands
        - execute_python_command(python_code: str): Python automation
        - API Calls: {config["executor"]["name"]}, execute_python_command

        IMPORTANT: Focus on BACKGROUND desktop environment manipulation that won't interfere with the main task:
        - System notifications, background processes, desktop themes
        - Window management of OTHER applications (not the main task)
        - Background file operations, system settings changes
        - Desktop environment modifications that don't affect the primary workflow

        {common_sections}

        Return JSON array with a list of exactly 2 scenario objects.
        """

        scenarios_data = self.call_llm(prompt)
        return self._parse_scenarios(scenarios_data, "terminal")

    def _generate_vs_code_scenarios(
        self, seed_trajectory: SeedTrajectory, app_state: Dict[str, Any], curriculum_config: CurriculumConfig
    ) -> List[ScenarioSpec]:
        """Generate VS Code-specific scenarios using Python automation"""

        config = self._get_app_specific_config("vs_code")
        common_sections = self._build_common_prompt_sections(
            "vs_code", config["executor"], config["scenario_types"], config["examples"]
        )

        prompt = f"""
        Generate VS Code perturbation scenarios for this GUI task:

        Task: {seed_trajectory.task_instruction}
        App State: {app_state}

        Generate 2 scenario specifications for VS Code manipulation using Python automation:

        AVAILABLE EXECUTORS:
        - {config["executor"]["name"]}({config["executor"]["language"].lower()}_code: str): {config["executor"]["language"]} automation
        - execute_bash_command(command: str): Raw bash commands
        - API Calls: {config["executor"]["name"]}, execute_bash_command

        IMPORTANT: Focus on BACKGROUND VS Code environment manipulation that won't interfere with the main task:
        - Background file operations, temporary file creation
        - Window management, panel resizing, sidebar toggling
        - System notifications, background processes
        - Desktop environment modifications that don't affect the primary workflow

        {common_sections}

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

    def _build_common_perturbation_prompt_sections(
        self, app_name: str, executor_info: Dict[str, str], focus_areas: List[str], examples: List[str]
    ) -> str:
        """Build enhanced perturbation decision prompts with app state awareness and systematic coverage"""

        current_state = """
        CURRENT STATE:
        - Step: {step_idx}
        - Action: {current_action}
        - Action History: {action_history}
        - CoT Context: {cot_context}
        - App States: {app_states}
        - Task: {task_instruction}
        """

        scenario_spec = """
        SCENARIO SPEC:
        - Target App: {target_app}
        - Trigger: {perturbation_trigger}
        - Available Actions: {available_perturbation_actions}
        - Learning Objectives: {learning_objectives}
        - Target Components: {target_components}
        - Perturbation Types: {perturbation_types}
        """

        # NEW: App state extraction and usage instructions
        app_state_usage = """
        ═══════════════════════════════════════════════════════════════
        📊 EXTRACTED APP STATE DATA - USE THIS INFORMATION!
        ═══════════════════════════════════════════════════════════════

        You have access to rich, structured app state information in {app_states}.

        🌐 BROWSER APP STATES CONTAIN:
        - buttons: Array of {{"id": "btn-id", "class": "btn-class", "text": "Save", "aria_label": "..."}}
        - links: Array of {{"href": "/path", "text": "Link", "id": "link-id"}}
        - input_fields: Array of {{"name": "email", "type": "text", "placeholder": "..."}}
        - forms: Array with field structures
        - headings: Content hierarchy
        - page_url, page_title: Context
        - interactive_elements_summary: Element counts

        📊 LIBREOFFICE APP STATES CONTAIN:
        - document_state: {{"sheets": [...], "active_sheet": "Sheet1", "sample_cells": [...]}}
        - buttons: Toolbar buttons with names
        - menus: Menu structure
        - text_fields: Formula bar, name box

        🖥️ ALL APPS CONTAIN:
        - interactive_elements: Up to 50 elements with positions
        - ui_structure: {{"has_menu_bar": true, "has_toolbar": true, ...}}
        - summary: Element counts and statistics

        ⚠️ CRITICAL: Generate code that TARGETS SPECIFIC ELEMENTS from app_states!
        ✅ GOOD: document.querySelector('#{{button.id}}') using real IDs from app_states
        ❌ BAD: document.querySelectorAll('button') using generic selectors
        """

        # NEW: 8 Visual Dimensions Framework
        visual_dimensions = """
        ═══════════════════════════════════════════════════════════════
        🎨 8-DIMENSIONAL VISUAL FEATURE SPACE - TARGET 2-3 DIMENSIONS
        ═══════════════════════════════════════════════════════════════

        1. COLOR: Backgrounds, text, borders, accents (use design system palettes)
        2. TYPOGRAPHY: Font families, sizes (12-24px), weights (300-900), spacing
        3. LAYOUT: Padding (4-48px), margins, alignment, container widths
        4. SHAPE: Border radius (0-24px), shadows (0-4 levels), border styles
        5. MOTION: Transitions (100-500ms), timing functions (keep subtle)
        6. DEPTH: Z-index, overlay opacity, shadow intensity
        7. DENSITY: Compact (4-8px padding) vs Spacious (20-32px padding)
        8. SEMANTICS: Primary/secondary styling, success/error colors

        🎯 SELECTION STRATEGY:
        - Choose 2-3 dimensions for MAXIMUM visual impact
        - Combine related dimensions (Color + Typography + Shape)
        - Avoid single-dimension changes (insufficient impact)
        - Target 4+ dimensions for EXTREME transformations

        ⚠️ TARGET HIGH IMPACT: Affect 15+ elements across 3+ dimensions!
        """

        # NEW: Design System Library
        design_systems = """
        ═══════════════════════════════════════════════════════════════
        🎨 PROFESSIONAL DESIGN SYSTEMS - USE THESE REAL PALETTES
        ═══════════════════════════════════════════════════════════════

        MATERIAL DESIGN 3: {{primary: '#6750A4', secondary: '#625B71', surface: '#FFFBFE',
                            outline: '#79747E'}}, Font: 'Roboto', Radius: 4-20px

        FLUENT: {{themePrimary: '#0078D4', themeDark: '#005A9E', neutralLight: '#F3F2F1'}},
                Font: 'Segoe UI', Radius: 0-8px

        APPLE HIG: {{systemBlue: '#007AFF', systemGreen: '#34C759', systemRed: '#FF3B30'}},
                   Font: 'SF Pro', Radius: 4-16px or 50%

        ANT DESIGN: {{blue: '#1890FF', green: '#52C41A', red: '#F5222D', gold: '#FAAD14'}},
                    Font: 'Roboto', Radius: 2-8px

        HIGH CONTRAST: {{bg: '#000000', fg: '#FFFFFF', link: '#FFFF00', focus: '#00FFFF'}},
                       Font: 'Arial', Radius: 0px (sharp), Borders: 2-4px thick

        💡 Pick ONE design system per perturbation and apply CONSISTENTLY!
        ⚠️ Maintain accessibility: text contrast ≥ 4.5:1, touch targets ≥ 44px
        """

        # NEW: Multi-Layer Orchestration
        multi_layer = """
        ═══════════════════════════════════════════════════════════════
        🎭 MULTI-LAYER PERTURBATION (Use 2+ layers for maximum impact)
        ═══════════════════════════════════════════════════════════════

        LAYER 1 (App-Level): execute_js_on_page() or execute_uno_command()
        LAYER 2 (System): execute_system_perturbation("desktop_theme", {{theme, icon_theme, color_scheme}})
        LAYER 3 (Shell): execute_bash_command("gsettings set ...", "wmctrl -r ...", "notify-send ...")
        LAYER 4 (Python): execute_python_command("import subprocess; ...")

        💡 COMBINATION EXAMPLES:
        - Complete Theme: App styling + System dark mode + Font change
        - High Contrast: App colors + HighContrast theme + Bold fonts
        - Layout Density: App padding + Normal theme + Window resize
        """

        # NEW: Diversity Scoring
        diversity_scoring = """
        ═══════════════════════════════════════════════════════════════
        📊 DIVERSITY SCORING - AIM FOR 35+/50 POINTS
        ═══════════════════════════════════════════════════════════════

        1. VISUAL IMPACT (10pts): 7-10 = Colors + Fonts + Layout + Shape transformation
        2. ELEMENT COVERAGE (10pts): 7-10 = 16+ elements modified
        3. DIMENSION DIVERSITY (10pts): 7-10 = 3-5 visual dimensions targeted
        4. REALISM (10pts): 7-10 = Professional design system, coherent, accessible
        5. ORIGINALITY (10pts): 7-10 = Completely novel, not copied from examples

        QUALITY TIERS:
        • 40-50: EXCELLENT - Maximum training value ⭐⭐⭐
        • 30-39: GOOD - Acceptable diversity ⭐⭐
        • 20-29: FAIR - Needs improvement ⭐
        • 0-19: POOR - Insufficient diversity ❌

        ⚠️ Self-evaluate BEFORE generating: Will this score 35+ points?
        """

        executor_info_section = f"""
        AVAILABLE EXECUTOR: {executor_info["name"]}({executor_info["language"].lower()}_code: str, parameters: Dict)
        - Input: Raw {executor_info["language"]} code (NO markdown, NO ```, NO language tags)
        - Use: {executor_info["description"]}
        - CRITICAL: Only modify visual elements, never interfere with main task functionality
        """

        decision_criteria = """
        DECISION CRITERIA:
        1. Does the current step match the perturbation trigger conditions?
        2. Is the target app active and relevant to the current action?
        3. What COMPLETE DESIGN SYSTEM transformation should be applied?
        4. Can I target SPECIFIC elements from app_states (buttons[], links[], inputs[])?
        5. Will this affect 15+ elements across 3+ visual dimensions?
        6. Does this use a REAL design system (Material/Fluent/HIG/Ant/HighContrast)?
        7. Is this ORIGINAL and sophisticated (not copying examples)?
        8. Will this score 35+ points on the diversity rubric?
        """

        focus_areas_section = f"        VISUAL INVARIANCE LEARNING FOCUS ({app_name.title()}-Specific):\n"
        for area in focus_areas:
            focus_areas_section += f"        - {area}\n"

        json_format = f"""
        REQUIRED JSON FORMAT (exactly these fields):
        {{{{
            "should_apply": true/false,
            "perturbation_type": "complete_design_system_transformation",
            "target_app": "{app_name}",
            "reasoning": "Which design system, why now, which elements from app_states",
            "generated_code": "COMPLETE_DESIGN_SYSTEM_RAW_{executor_info["language"].upper()}_CODE_TARGETING_SPECIFIC_ELEMENTS_FROM_APP_STATES_AFFECTING_15+_ELEMENTS_ACROSS_3+_DIMENSIONS",
            "api_call": "{executor_info["name"]}",
            "parameters": {{{{"target_app": "{app_name}"}}}}
        }}}}
        """

        examples_section = (
            f"        DESIGN SYSTEM EXAMPLES ({app_name}) - FOR INSPIRATION ONLY, CREATE YOUR OWN:\n"
        )
        for example in examples:
            examples_section += f"        {example}\n"

        creativity_instructions = """
        ═══════════════════════════════════════════════════════════════
        🚀 CREATIVITY & ORIGINALITY REQUIREMENTS
        ═══════════════════════════════════════════════════════════════

        ❌ DO NOT:
        - Copy examples verbatim
        - Use generic selectors like querySelectorAll('button')
        - Change only 1-2 elements or 1 dimension
        - Use random colors without a design system
        - Score below 35/50 on diversity rubric

        ✅ DO:
        - Reference SPECIFIC elements from app_states (buttons[0].id, links[1].href)
        - Apply COMPLETE design system (Material/Fluent/HIG/Ant/HighContrast)
        - Target 15+ elements with coordinated styling
        - Combine 3+ dimensions (Color + Typography + Layout/Shape)
        - Create NOVEL combinations using sophisticated logic
        - Think: "Would this look like a real product redesign?"

        💡 SOPHISTICATION EXAMPLES:
        - Loop through app_states.buttons and apply primary/secondary colors alternately
        - Match font system to design system (Roboto for Material, Segoe UI for Fluent)
        - Apply elevation shadows based on element hierarchy
        - Create density variations (compact vs spacious modes)
        - Combine app-level styling + system theme changes
        """

        return f"""
        {current_state}

        {scenario_spec}

        {app_state_usage}

        {visual_dimensions}

        {design_systems}

        {multi_layer}

        {diversity_scoring}

        {executor_info_section}

        {decision_criteria}

        {focus_areas_section}

        {json_format}

        {examples_section}

        {creativity_instructions}
        """

    def _get_perturbation_app_config(self, app_name: str) -> Dict[str, Any]:
        """Get app-specific configuration for perturbation decision prompts"""
        configs = {
            "browser": {
                "executor": {
                    "name": "execute_js_on_page",
                    "language": "JavaScript",
                    "description": "Complete design system transformations targeting specific elements from app_states",
                },
                "focus_areas": [
                    "Apply COMPLETE DESIGN SYSTEM transformations (Material 3, Fluent, HIG, Ant, HighContrast)",
                    "Target SPECIFIC elements from app_states.buttons[], links[], input_fields[], forms[]",
                    "Modify 3+ visual dimensions: Color + Typography + Layout/Shape combined",
                    "Use REAL professional color palettes from established design systems",
                    "Create DRASTICALLY DIFFERENT appearances affecting 15+ UI elements",
                    "Maintain functionality while achieving maximum visual diversity",
                ],
                "examples": [
                    """
                    // Material Design 3 Complete Transformation (Score: 45/50)
                    const btns = {app_states}.buttons || [];
                    const links = {app_states}.links || [];
                    const inputs = {app_states}.input_fields || [];
                    const md3 = {primary: '#6750A4', onPrimary: '#FFFFFF', secondary: '#625B71',
                                onSecondary: '#FFFFFF', tertiary: '#7D5260', surface: '#FFFBFE',
                                onSurface: '#1C1B1F', outline: '#79747E'};
                    btns.forEach((b, i) => {
                      const el = document.querySelector(`#${b.id}`) || document.querySelector(`.${b.class?.split(' ')[0]}`);
                      if (el) {
                        el.style.backgroundColor = i % 3 === 0 ? md3.primary : i % 3 === 1 ? md3.secondary : md3.tertiary;
                        el.style.color = i % 3 === 0 ? md3.onPrimary : md3.onSecondary;
                        el.style.fontFamily = "'Roboto', sans-serif";
                        el.style.fontSize = '14px';
                        el.style.fontWeight = '500';
                        el.style.borderRadius = '20px';
                        el.style.padding = '10px 24px';
                        el.style.border = 'none';
                        el.style.boxShadow = '0 1px 2px rgba(0,0,0,0.3), 0 1px 3px rgba(0,0,0,0.15)';
                        el.style.transition = 'all 200ms cubic-bezier(0.4, 0.0, 0.2, 1)';
                      }
                    });
                    links.forEach(l => {
                      const el = document.querySelector(`a[href='${l.href}']`);
                      if (el) { el.style.color = md3.tertiary; el.style.textDecoration = 'none'; el.style.fontWeight = '500'; }
                    });
                    inputs.forEach(inp => {
                      const el = document.querySelector(`input[name='${inp.name}']`);
                      if (el) {
                        el.style.backgroundColor = md3.surface;
                        el.style.color = md3.onSurface;
                        el.style.border = `1px solid ${md3.outline}`;
                        el.style.borderRadius = '4px';
                        el.style.padding = '16px';
                        el.style.fontFamily = "'Roboto', sans-serif";
                        el.style.fontSize = '16px';
                      }
                    });
                    document.body.style.backgroundColor = md3.surface;
                    document.body.style.color = md3.onSurface;
                    document.querySelectorAll('.container, .card, section, article').forEach(el => {
                      el.style.backgroundColor = '#FFFFFF';
                      el.style.borderRadius = '12px';
                      el.style.padding = '16px';
                      el.style.boxShadow = '0 1px 3px rgba(0,0,0,0.12)';
                    });
                    """,
                    """
                    // Fluent Design Light Theme (Score: 43/50)
                    const fluent = {themePrimary: '#0078D4', themeDark: '#005A9E', neutralLight: '#F3F2F1',
                                   neutralDark: '#201f1e', white: '#FFFFFF'};
                    const allInteractive = [...({app_states}.buttons || []), ...({app_states}.links || [])];
                    allInteractive.forEach((item, idx) => {
                      let el = item.id ? document.querySelector(`#${item.id}`) :
                               item.class ? document.querySelector(`.${item.class.split(' ')[0]}`) :
                               item.href ? document.querySelector(`a[href='${item.href}']`) : null;
                      if (el) {
                        el.style.fontFamily = "'Segoe UI', system-ui, sans-serif";
                        el.style.fontSize = '14px';
                        if (el.tagName === 'BUTTON' || el.tagName === 'INPUT') {
                          el.style.backgroundColor = idx % 2 === 0 ? fluent.themePrimary : fluent.white;
                          el.style.color = idx % 2 === 0 ? fluent.white : fluent.themeDark;
                          el.style.border = `1px solid ${fluent.neutralLight}`;
                          el.style.borderRadius = '2px';
                          el.style.padding = '8px 16px';
                          el.style.boxShadow = '0 3.2px 7.2px rgba(0,0,0,0.13)';
                        } else {
                          el.style.color = fluent.themePrimary;
                          el.style.fontWeight = '600';
                        }
                        el.style.transition = 'all 150ms ease';
                      }
                    });
                    ({app_states}.forms || []).forEach(f => {
                      const el = document.querySelector(`#${f.id}`);
                      if (el) {
                        el.style.padding = '24px';
                        el.style.backgroundColor = fluent.white;
                        el.style.borderRadius = '2px';
                        el.style.boxShadow = '0 3.2px 7.2px rgba(0,0,0,0.13), 0 0.6px 1.8px rgba(0,0,0,0.11)';
                      }
                    });
                    document.body.style.backgroundColor = fluent.neutralLight;
                    document.body.style.color = fluent.neutralDark;
                    """,
                    """
                    // High Contrast Accessibility Theme (Score: 42/50)
                    const hc = {bg: '#000000', fg: '#FFFFFF', link: '#FFFF00', button: '#FFFFFF', focus: '#00FFFF'};
                    document.body.style.backgroundColor = hc.bg;
                    document.body.style.color = hc.fg;
                    document.body.style.fontFamily = "'Arial', sans-serif";
                    document.body.style.fontSize = '16px';
                    const allEls = [...({app_states}.buttons || []), ...({app_states}.links || []),
                                    ...({app_states}.input_fields || [])];
                    allEls.forEach(item => {
                      let el = item.id ? document.querySelector(`#${item.id}`) :
                               item.class ? document.querySelector(`.${item.class.split(' ')[0]}`) :
                               item.name ? document.querySelector(`[name='${item.name}']`) :
                               item.href ? document.querySelector(`a[href='${item.href}']`) : null;
                      if (el) {
                        el.style.backgroundColor = hc.bg;
                        el.style.color = el.tagName === 'A' ? hc.link : hc.fg;
                        el.style.border = `3px solid ${hc.button}`;
                        el.style.borderRadius = '0';
                        el.style.padding = '12px 24px';
                        el.style.fontFamily = "'Arial Black', sans-serif";
                        el.style.fontSize = '16px';
                        el.style.fontWeight = '700';
                        el.style.outline = `3px solid ${hc.focus}`;
                        el.style.outlineOffset = '2px';
                        if (el.tagName === 'A') {
                          el.style.textDecoration = 'underline';
                          el.style.textUnderlineOffset = '4px';
                        }
                      }
                    });
                    document.querySelectorAll('h1, h2, h3, h4, h5, h6').forEach(h => {
                      h.style.color = hc.fg;
                      h.style.fontWeight = '900';
                      h.style.borderBottom = `2px solid ${hc.fg}`;
                      h.style.paddingBottom = '8px';
                    });
                    document.querySelectorAll('div, section, article').forEach(el => {
                      el.style.border = `1px solid ${hc.fg}`;
                      el.style.borderRadius = '0';
                    });
                    """,
                ],
            },
            "libreoffice_calc": {
                "executor": {
                    "name": "execute_uno_command",
                    "language": "UNO Python",
                    "description": "Complete spreadsheet visual themes using UNO API with app_states awareness",
                },
                "focus_areas": [
                    "Use document_state from app_states to target specific sheets/cells",
                    "Apply COMPLETE visual theme transformations (Finance, Marketing, Engineering themes)",
                    "Modify colors + fonts + borders systematically across grid",
                    "Change grid, toolbar, and zoom settings for visual variations",
                    "Affect 20+ cells with professional color schemes",
                    "NEVER touch spreadsheet data, calculations, or formulas",
                ],
                "examples": [
                    """
                    # Professional Finance Theme (Score: 40/50)
                    doc = desktop.getCurrentComponent()
                    sheets = doc.getSheets()
                    # Use active sheet from app_states
                    active_sheet_name = {app_states}.get('document_state', {}).get('active_sheet', 'Sheet1')
                    sheet = sheets.getByName(active_sheet_name) if active_sheet_name in [sheets.getByIndex(i).getName() for i in range(sheets.getCount())] else sheets.getByIndex(0)
                    # Finance color palette
                    headerBg, headerText = 0x0F4C81, 0xFFFFFF
                    altRowBg, borderColor = 0xF0F4F8, 0xCDD5DE
                    # Style header row with deep blue
                    for col in range(10):
                        cell = sheet.getCellByPosition(col, 0)
                        cell.CellBackColor = headerBg
                        cell.CharColor = headerText
                        cell.CharWeight = 150
                        cell.CharHeight = 11
                        cell.CharFontName = "Calibri"
                        from com.sun.star.table import BorderLine2
                        border = BorderLine2()
                        border.Color = borderColor
                        border.OuterLineWidth = 20
                        cell.TopBorder = border
                        cell.BottomBorder = border
                    # Alternating row colors for better readability
                    for row in range(1, 50):
                        for col in range(10):
                            cell = sheet.getCellByPosition(col, row)
                            if row % 2 == 0:
                                cell.CellBackColor = altRowBg
                            cell.CharHeight = 10
                            cell.CharFontName = "Calibri"
                    # Grid settings
                    viewSettings = doc.getCurrentController().getViewSettings()
                    viewSettings.setPropertyValue('ShowGrid', True)
                    viewSettings.setPropertyValue('GridColor', borderColor)
                    viewSettings.setPropertyValue('ZoomValue', 100)
                    """,
                    """
                    # Marketing Theme - Vibrant Colors (Score: 38/50)
                    doc = desktop.getCurrentComponent()
                    sheet = doc.getSheets().getByIndex(0)
                    # Marketing color palette - vibrant and engaging
                    brandPrimary, brandSecondary = 0xFF6B35, 0x004E89
                    accentGreen, neutralBg = 0x2DD881, 0xFAFAFA
                    # Bold header styling
                    for col in range(8):
                        cell = sheet.getCellByPosition(col, 0)
                        cell.CellBackColor = brandPrimary
                        cell.CharColor = 0xFFFFFF
                        cell.CharWeight = 150
                        cell.CharHeight = 12
                        cell.CharFontName = "Arial"
                        cell.HoriJustify = 2  # Center
                    # Data cells with modern styling
                    for row in range(1, 30):
                        for col in range(8):
                            cell = sheet.getCellByPosition(col, row)
                            cell.CellBackColor = neutralBg
                            cell.CharHeight = 10
                            cell.CharFontName = "Arial"
                            if col == 0:  # Accent first column
                                cell.CharColor = brandSecondary
                                cell.CharWeight = 150
                    # View settings
                    viewSettings = doc.getCurrentController().getViewSettings()
                    viewSettings.setPropertyValue('ZoomValue', 120)
                    viewSettings.setPropertyValue('ShowGrid', False)
                    """,
                    """
                    # High Contrast Engineering Theme (Score: 37/50)
                    doc = desktop.getCurrentComponent()
                    sheets = doc.getSheets()
                    sheet = sheets.getByIndex(0)
                    # High contrast colors for engineering precision
                    darkBg, lightText = 0x1E1E1E, 0xFFFFFF
                    accentOrange, gridColor = 0xFF9500, 0x3C3C3C
                    # Dark theme headers
                    for col in range(12):
                        cell = sheet.getCellByPosition(col, 0)
                        cell.CellBackColor = darkBg
                        cell.CharColor = lightText
                        cell.CharWeight = 150
                        cell.CharHeight = 10
                        cell.CharFontName = "Courier New"
                        from com.sun.star.table import BorderLine2
                        border = BorderLine2()
                        border.Color = gridColor
                        border.OuterLineWidth = 15
                        cell.BottomBorder = border
                    # Data cells - monospace for precision
                    for row in range(1, 40):
                        for col in range(12):
                            cell = sheet.getCellByPosition(col, row)
                            cell.CellBackColor = 0x2D2D2D if row % 2 == 0 else 0x252525
                            cell.CharColor = lightText
                            cell.CharHeight = 9
                            cell.CharFontName = "Courier New"
                            if col == 0:
                                cell.CharColor = accentOrange
                    # View settings for engineering work
                    viewSettings = doc.getCurrentController().getViewSettings()
                    viewSettings.setPropertyValue('ShowGrid', True)
                    viewSettings.setPropertyValue('GridColor', gridColor)
                    viewSettings.setPropertyValue('ZoomValue', 110)
                    """,
                ],
            },
            "libreoffice_impress": {
                "executor": {
                    "name": "execute_uno_command",
                    "language": "UNO Python",
                    "description": "Complete presentation visual themes using UNO API with app_states awareness",
                },
                "focus_areas": [
                    "Use document_state from app_states to target specific slides",
                    "Apply COMPLETE presentation themes (Corporate, Academic, Creative themes)",
                    "Modify slide backgrounds + view modes + zoom systematically",
                    "Change colors, fonts, layout styles for visual variations",
                    "Affect multiple slides with professional design systems",
                    "NEVER touch slide content, text, or presentation structure",
                ],
                "examples": [
                    """
                    # Corporate Professional Theme (Score: 38/50)
                    doc = desktop.getCurrentComponent()
                    slides = doc.getDrawPages()
                    slide_count = slides.getCount()
                    # Corporate color palette - professional blue
                    corporateBg, accentBlue = 0xF8F9FA, 0x0066CC
                    titleBg = 0xE8F0FE
                    # Apply theme to all slides
                    for i in range(min(slide_count, 10)):
                        slide = slides.getByIndex(i)
                        slide.setPropertyValue('BackgroundColor', corporateBg if i > 0 else titleBg)
                    # View settings for professional presentation
                    viewSettings = doc.getCurrentController().getViewSettings()
                    viewSettings.setPropertyValue('ZoomType', 1)  # Optimal zoom
                    viewSettings.setPropertyValue('ShowRuler', False)
                    viewSettings.setPropertyValue('ShowSlidePane', True)
                    """,
                    """
                    # Academic Research Theme (Score: 36/50)
                    doc = desktop.getCurrentComponent()
                    slides = doc.getDrawPages()
                    # Academic color palette - neutral with green accents
                    neutralBg, accentGreen = 0xFFFBF5, 0x2E7D32
                    headerBg = 0xE8F5E9
                    # Theme first 5 slides
                    for i in range(min(slides.getCount(), 5)):
                        slide = slides.getByIndex(i)
                        slide.setPropertyValue('BackgroundColor', headerBg if i == 0 else neutralBg)
                    # Academic view settings
                    viewSettings = doc.getCurrentController().getViewSettings()
                    viewSettings.setPropertyValue('ZoomType', 0)  # Page view
                    viewSettings.setPropertyValue('ShowNotesPane', True)
                    """,
                    """
                    # Creative Design Theme (Score: 37/50)
                    doc = desktop.getCurrentComponent()
                    slides = doc.getDrawPages()
                    # Creative color palette - vibrant gradients
                    vibrantColors = [0xFFE5E5, 0xE5F5FF, 0xFFF5E5, 0xF0E5FF, 0xE5FFE5]
                    for i in range(min(slides.getCount(), len(vibrantColors))):
                        slide = slides.getByIndex(i)
                        slide.setPropertyValue('BackgroundColor', vibrantColors[i])
                    # Creative view settings
                    viewSettings = doc.getCurrentController().getViewSettings()
                    viewSettings.setPropertyValue('ZoomType', 2)  # Fit width
                    viewSettings.setPropertyValue('ShowSlideSorter', False)
                    """,
                ],
            },
            "libreoffice_writer": {
                "executor": {
                    "name": "execute_uno_command",
                    "language": "UNO Python",
                    "description": "Complete document visual themes using UNO API with app_states awareness",
                },
                "focus_areas": [
                    "Use document_state from app_states to apply context-aware themes",
                    "Apply COMPLETE document themes (Professional, Academic, Creative themes)",
                    "Modify page layouts + view settings + zoom systematically",
                    "Change colors, fonts, margins for visual variations",
                    "Affect document-wide appearance with design system consistency",
                    "NEVER touch document content, text, or document structure",
                ],
                "examples": [
                    """
                    # Professional Business Document Theme (Score: 37/50)
                    doc = desktop.getCurrentComponent()
                    # Professional page style
                    pageStyles = doc.getStyleFamilies().getByName('PageStyles')
                    standardPage = pageStyles.getByName('Standard')
                    standardPage.setPropertyValue('BackColor', 0xFFFFFF)
                    standardPage.setPropertyValue('HeaderIsOn', True)
                    standardPage.setPropertyValue('FooterIsOn', True)
                    standardPage.setPropertyValue('LeftMargin', 2500)
                    standardPage.setPropertyValue('RightMargin', 2500)
                    # View settings for professional work
                    viewSettings = doc.getCurrentController().getViewSettings()
                    viewSettings.setPropertyValue('ShowRuler', True)
                    viewSettings.setPropertyValue('ShowStatusBar', True)
                    viewSettings.setPropertyValue('ZoomType', 0)  # Optimal view
                    """,
                    """
                    # Academic Paper Theme (Score: 36/50)
                    doc = desktop.getCurrentComponent()
                    # Academic page style - wider margins
                    pageStyles = doc.getStyleFamilies().getByName('PageStyles')
                    standardPage = pageStyles.getByName('Standard')
                    standardPage.setPropertyValue('BackColor', 0xFFFBF5)
                    standardPage.setPropertyValue('LeftMargin', 3000)  # Wider for binding
                    standardPage.setPropertyValue('RightMargin', 2000)
                    standardPage.setPropertyValue('TopMargin', 2500)
                    standardPage.setPropertyValue('BottomMargin', 2500)
                    # Academic view settings
                    viewSettings = doc.getCurrentController().getViewSettings()
                    viewSettings.setPropertyValue('ShowRuler', True)
                    viewSettings.setPropertyValue('ShowTextBoundaries', True)
                    viewSettings.setPropertyValue('ZoomValue', 120)  # Larger for reading
                    """,
                    """
                    # Creative Writing Theme (Score: 35/50)
                    doc = desktop.getCurrentComponent()
                    # Creative page style - minimal distractions
                    pageStyles = doc.getStyleFamilies().getByName('PageStyles')
                    standardPage = pageStyles.getByName('Standard')
                    standardPage.setPropertyValue('BackColor', 0xFFF8E1)  # Warm background
                    standardPage.setPropertyValue('LeftMargin', 2000)
                    standardPage.setPropertyValue('RightMargin', 2000)
                    # Minimal view for focus
                    viewSettings = doc.getCurrentController().getViewSettings()
                    viewSettings.setPropertyValue('ShowRuler', False)
                    viewSettings.setPropertyValue('ShowStatusBar', False)
                    viewSettings.setPropertyValue('ShowTextBoundaries', False)
                    viewSettings.setPropertyValue('ZoomType', 3)  # Page width
                    """,
                ],
            },
            "gimp": {
                "executor": {
                    "name": "execute_bash_command",
                    "language": "bash",
                    "description": "System-level theme transformations and background processes",
                },
                "focus_areas": [
                    "Apply COMPLETE system theme transformations (Light/Dark/HighContrast)",
                    "Combine GTK theme + icon theme + font changes systematically",
                    "Add background notifications for realistic work environment simulation",
                    "NEVER interfere with main task image editing or file operations",
                ],
                "examples": [
                    """# Dark Theme Complete System (Score: 32/50)
                    THEMES=('Adwaita-dark' 'HighContrast'); ICONS=('Papirus-Dark' 'HighContrast'); THEME=${THEMES[$RANDOM % ${#THEMES[@]}]}; ICON=${ICONS[$RANDOM % ${#ICONS[@]}]}; gsettings set org.gnome.desktop.interface gtk-theme "$THEME"; gsettings set org.gnome.desktop.interface icon-theme "$ICON"; gsettings set org.gnome.desktop.interface color-scheme 'prefer-dark'; gsettings set org.gnome.desktop.interface font-name 'Ubuntu Bold 11'""",
                    """# Light Professional Theme (Score: 30/50)
                    gsettings set org.gnome.desktop.interface gtk-theme 'Adwaita'; gsettings set org.gnome.desktop.interface icon-theme 'Papirus'; gsettings set org.gnome.desktop.interface color-scheme 'prefer-light'; gsettings set org.gnome.desktop.interface font-name 'Ubuntu 11'; notify-send 'System' 'Theme updated' --icon=preferences-desktop-theme""",
                ],
            },
            "file_manager": {
                "executor": {
                    "name": "execute_bash_command",
                    "language": "bash",
                    "description": "System-level theme transformations and window management",
                },
                "focus_areas": [
                    "Apply system theme changes combined with window positioning",
                    "Add background file operations for realistic environment",
                    "Combine theme + icon + font + window management",
                    "NEVER interfere with main task file operations or navigation",
                ],
                "examples": [
                    """# Theme + Window Combo (Score: 31/50)
                    gsettings set org.gnome.desktop.interface gtk-theme 'Adwaita-dark'; gsettings set org.gnome.desktop.interface icon-theme 'Yaru-dark'; wmctrl -r 'Files' -e 0,100,100,1000,700 2>/dev/null || true; notify-send 'File Manager' 'Environment updated' --icon=folder""",
                ],
            },
            "terminal": {
                "executor": {
                    "name": "execute_bash_command",
                    "language": "bash",
                    "description": "Background system manipulations that don't affect terminal tasks",
                },
                "focus_areas": [
                    "BACKGROUND system theme changes only (don't affect terminal content)",
                    "Background notifications and processes",
                    "Desktop environment changes (not terminal-specific)",
                    "NEVER interfere with terminal commands or task execution",
                ],
                "examples": [
                    """# Background System Changes (Score: 28/50)
                    gsettings set org.gnome.desktop.interface gtk-theme 'Adwaita-dark'; notify-send 'Background' 'System maintenance' --icon=system-run; mkdir -p /tmp/bg_proc && echo 'process' > /tmp/bg_proc/log.txt""",
                ],
            },
            "vs_code": {
                "executor": {
                    "name": "execute_python_command",
                    "language": "Python",
                    "description": "Background system manipulations that don't affect VS Code tasks",
                },
                "focus_areas": [
                    "BACKGROUND file operations and system changes only",
                    "Desktop theme modifications (not VS Code-specific)",
                    "Background notifications and processes",
                    "NEVER interfere with VS Code editing or main task",
                ],
                "examples": [
                    """# Background System Operations (Score: 27/50)
                    import os, subprocess; os.makedirs('/tmp/bg_work', exist_ok=True); open('/tmp/bg_work/log.txt', 'w').write('Background process'); subprocess.run(['notify-send', 'Background', 'Process running'], check=False)""",
                ],
            },
            "system": {
                "executor": {
                    "name": "execute_python_command",
                    "language": "Python",
                    "description": "Background desktop environment manipulations",
                },
                "focus_areas": [
                    "BACKGROUND desktop theme changes only",
                    "System notifications and background processes",
                    "Desktop environment modifications (not app-specific)",
                    "NEVER interfere with the main task or application",
                ],
                "examples": [
                    """# Background Desktop Changes (Score: 26/50)
                    import subprocess, os; subprocess.run(['gsettings', 'set', 'org.gnome.desktop.interface', 'gtk-theme', 'Adwaita-dark'], check=False); subprocess.run(['notify-send', 'System', 'Background update'], check=False); os.makedirs('/tmp/sys_bg', exist_ok=True)""",
                ],
            },
        }
        return configs.get(app_name, configs["browser"])  # Default to browser config

    def decide_perturbation(
        self, execution_context: ExecutionContext, scenario_spec: ScenarioSpec
    ) -> Dict[str, Any]:
        """Decide whether to apply perturbation at current step"""

        # Route to app-specific perturbation decision
        target_app = scenario_spec.target_app.lower()

        if target_app == "browser":
            return self._decide_browser_perturbation(execution_context, scenario_spec)
        elif target_app == "libreoffice_calc":
            return self._decide_libreoffice_calc_perturbation(execution_context, scenario_spec)
        elif target_app == "libreoffice_impress":
            return self._decide_libreoffice_impress_perturbation(execution_context, scenario_spec)
        elif target_app == "libreoffice_writer":
            return self._decide_libreoffice_writer_perturbation(execution_context, scenario_spec)
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

        config = self._get_perturbation_app_config("browser")
        common_sections = self._build_common_perturbation_prompt_sections(
            "browser", config["executor"], config["focus_areas"], config["examples"]
        )

        prompt = f"""
        Decide whether to apply a sophisticated browser perturbation during GUI task execution.

        {
            common_sections.format(
                step_idx=execution_context.step_idx,
                current_action=execution_context.current_action,
                action_history=execution_context.action_history[-3:]
                if execution_context.action_history
                else [],
                cot_context=execution_context.cot_context,
                app_states=execution_context.app_states,
                task_instruction=execution_context.task_instruction,
                target_app=scenario_spec.target_app,
                perturbation_trigger=scenario_spec.perturbation_trigger,
                available_perturbation_actions=scenario_spec.available_perturbation_actions,
                learning_objectives=scenario_spec.learning_objectives,
                target_components=scenario_spec.target_components,
                perturbation_types=[pt.value for pt in scenario_spec.perturbation_types],
            )
        }

        Return JSON object with exactly the required fields.
        """

        response = self.call_llm(prompt)
        return self._validate_perturbation_decision(response)

    def _decide_libreoffice_calc_perturbation(
        self, execution_context: ExecutionContext, scenario_spec: ScenarioSpec
    ) -> Dict[str, Any]:
        """Decide LibreOffice Calc-specific perturbations using UNO commands"""

        config = self._get_perturbation_app_config("libreoffice_calc")
        common_sections = self._build_common_perturbation_prompt_sections(
            "libreoffice_calc", config["executor"], config["focus_areas"], config["examples"]
        )

        prompt = f"""
        Decide whether to apply a sophisticated LibreOffice Calc perturbation during GUI task execution.

        {
            common_sections.format(
                step_idx=execution_context.step_idx,
                current_action=execution_context.current_action,
                action_history=execution_context.action_history[-3:]
                if execution_context.action_history
                else [],
                cot_context=execution_context.cot_context,
                app_states=execution_context.app_states,
                task_instruction=execution_context.task_instruction,
                target_app=scenario_spec.target_app,
                perturbation_trigger=scenario_spec.perturbation_trigger,
                available_perturbation_actions=scenario_spec.available_perturbation_actions,
                learning_objectives=scenario_spec.learning_objectives,
                target_components=scenario_spec.target_components,
                perturbation_types=[pt.value for pt in scenario_spec.perturbation_types],
            )
        }

        Return JSON object with exactly the required fields.
        """

        response = self.call_llm(prompt)
        return self._validate_perturbation_decision(response)

    def _decide_libreoffice_impress_perturbation(
        self, execution_context: ExecutionContext, scenario_spec: ScenarioSpec
    ) -> Dict[str, Any]:
        """Decide LibreOffice Impress-specific perturbations using UNO commands"""

        config = self._get_perturbation_app_config("libreoffice_impress")
        common_sections = self._build_common_perturbation_prompt_sections(
            "libreoffice_impress", config["executor"], config["focus_areas"], config["examples"]
        )

        prompt = f"""
        Decide whether to apply a sophisticated LibreOffice Impress perturbation during GUI task execution.

        {
            common_sections.format(
                step_idx=execution_context.step_idx,
                current_action=execution_context.current_action,
                action_history=execution_context.action_history[-3:]
                if execution_context.action_history
                else [],
                cot_context=execution_context.cot_context,
                app_states=execution_context.app_states,
                task_instruction=execution_context.task_instruction,
                target_app=scenario_spec.target_app,
                perturbation_trigger=scenario_spec.perturbation_trigger,
                available_perturbation_actions=scenario_spec.available_perturbation_actions,
                learning_objectives=scenario_spec.learning_objectives,
                target_components=scenario_spec.target_components,
                perturbation_types=[pt.value for pt in scenario_spec.perturbation_types],
            )
        }

        Return JSON object with exactly the required fields.
        """

        response = self.call_llm(prompt)
        return self._validate_perturbation_decision(response)

    def _decide_libreoffice_writer_perturbation(
        self, execution_context: ExecutionContext, scenario_spec: ScenarioSpec
    ) -> Dict[str, Any]:
        """Decide LibreOffice Writer-specific perturbations using UNO commands"""

        config = self._get_perturbation_app_config("libreoffice_writer")
        common_sections = self._build_common_perturbation_prompt_sections(
            "libreoffice_writer", config["executor"], config["focus_areas"], config["examples"]
        )

        prompt = f"""
        Decide whether to apply a sophisticated LibreOffice Writer perturbation during GUI task execution.

        {
            common_sections.format(
                step_idx=execution_context.step_idx,
                current_action=execution_context.current_action,
                action_history=execution_context.action_history[-3:]
                if execution_context.action_history
                else [],
                cot_context=execution_context.cot_context,
                app_states=execution_context.app_states,
                task_instruction=execution_context.task_instruction,
                target_app=scenario_spec.target_app,
                perturbation_trigger=scenario_spec.perturbation_trigger,
                available_perturbation_actions=scenario_spec.available_perturbation_actions,
                learning_objectives=scenario_spec.learning_objectives,
                target_components=scenario_spec.target_components,
                perturbation_types=[pt.value for pt in scenario_spec.perturbation_types],
            )
        }

        Return JSON object with exactly the required fields.
        """

        response = self.call_llm(prompt)
        return self._validate_perturbation_decision(response)

    def _decide_image_editor_perturbation(
        self, execution_context: ExecutionContext, scenario_spec: ScenarioSpec
    ) -> Dict[str, Any]:
        """Decide image editor perturbations using bash commands and app state manipulation"""

        config = self._get_perturbation_app_config("gimp")
        common_sections = self._build_common_perturbation_prompt_sections(
            "gimp", config["executor"], config["focus_areas"], config["examples"]
        )

        prompt = f"""
        Decide whether to apply an image editor perturbation during GUI task execution.

        {
            common_sections.format(
                step_idx=execution_context.step_idx,
                current_action=execution_context.current_action,
                action_history=execution_context.action_history[-3:]
                if execution_context.action_history
                else [],
                cot_context=execution_context.cot_context,
                app_states=execution_context.app_states,
                task_instruction=execution_context.task_instruction,
                target_app=scenario_spec.target_app,
                perturbation_trigger=scenario_spec.perturbation_trigger,
                available_perturbation_actions=scenario_spec.available_perturbation_actions,
                learning_objectives=scenario_spec.learning_objectives,
                target_components=scenario_spec.target_components,
                perturbation_types=[pt.value for pt in scenario_spec.perturbation_types],
            )
        }

        Return JSON object with exactly the required fields.
        """

        response = self.call_llm(prompt)
        return self._validate_perturbation_decision(response)

    def _decide_file_manager_perturbation(
        self, execution_context: ExecutionContext, scenario_spec: ScenarioSpec
    ) -> Dict[str, Any]:
        """Decide file manager perturbations using bash and Python commands"""

        config = self._get_perturbation_app_config("file_manager")
        common_sections = self._build_common_perturbation_prompt_sections(
            "file_manager", config["executor"], config["focus_areas"], config["examples"]
        )

        prompt = f"""
        Decide whether to apply a file manager perturbation during GUI task execution.

        {
            common_sections.format(
                step_idx=execution_context.step_idx,
                current_action=execution_context.current_action,
                action_history=execution_context.action_history[-3:]
                if execution_context.action_history
                else [],
                cot_context=execution_context.cot_context,
                app_states=execution_context.app_states,
                task_instruction=execution_context.task_instruction,
                target_app=scenario_spec.target_app,
                perturbation_trigger=scenario_spec.perturbation_trigger,
                available_perturbation_actions=scenario_spec.available_perturbation_actions,
                learning_objectives=scenario_spec.learning_objectives,
                target_components=scenario_spec.target_components,
                perturbation_types=[pt.value for pt in scenario_spec.perturbation_types],
            )
        }

        Return JSON object with exactly the required fields.
        """

        response = self.call_llm(prompt)
        return self._validate_perturbation_decision(response)

    def _decide_generic_perturbation(
        self, execution_context: ExecutionContext, scenario_spec: ScenarioSpec
    ) -> Dict[str, Any]:
        """Decide generic perturbations using Python commands"""

        config = self._get_perturbation_app_config("system")
        common_sections = self._build_common_perturbation_prompt_sections(
            "system", config["executor"], config["focus_areas"], config["examples"]
        )

        prompt = f"""
        Decide whether to apply a generic perturbation during GUI task execution.

        {
            common_sections.format(
                step_idx=execution_context.step_idx,
                current_action=execution_context.current_action,
                action_history=execution_context.action_history[-3:]
                if execution_context.action_history
                else [],
                cot_context=execution_context.cot_context,
                app_states=execution_context.app_states,
                task_instruction=execution_context.task_instruction,
                target_app=scenario_spec.target_app,
                perturbation_trigger=scenario_spec.perturbation_trigger,
                available_perturbation_actions=scenario_spec.available_perturbation_actions,
                learning_objectives=scenario_spec.learning_objectives,
                target_components=scenario_spec.target_components,
                perturbation_types=[pt.value for pt in scenario_spec.perturbation_types],
            )
        }

        Return JSON object with exactly the required fields.
        """

        response = self.call_llm(prompt)
        return self._validate_perturbation_decision(response)

    def _decide_terminal_perturbation(
        self, execution_context: ExecutionContext, scenario_spec: ScenarioSpec
    ) -> Dict[str, Any]:
        """Decide terminal perturbations using bash and Python commands"""

        config = self._get_perturbation_app_config("terminal")
        common_sections = self._build_common_perturbation_prompt_sections(
            "terminal", config["executor"], config["focus_areas"], config["examples"]
        )

        prompt = f"""
        Decide whether to apply a terminal perturbation during GUI task execution.

        {
            common_sections.format(
                step_idx=execution_context.step_idx,
                current_action=execution_context.current_action,
                action_history=execution_context.action_history[-3:]
                if execution_context.action_history
                else [],
                cot_context=execution_context.cot_context,
                app_states=execution_context.app_states,
                task_instruction=execution_context.task_instruction,
                target_app=scenario_spec.target_app,
                perturbation_trigger=scenario_spec.perturbation_trigger,
                available_perturbation_actions=scenario_spec.available_perturbation_actions,
                learning_objectives=scenario_spec.learning_objectives,
                target_components=scenario_spec.target_components,
                perturbation_types=[pt.value for pt in scenario_spec.perturbation_types],
            )
        }

        Return JSON object with exactly the required fields.
        """

        response = self.call_llm(prompt)
        return self._validate_perturbation_decision(response)

    def _decide_vs_code_perturbation(
        self, execution_context: ExecutionContext, scenario_spec: ScenarioSpec
    ) -> Dict[str, Any]:
        """Decide VS Code perturbations using Python automation and bash commands"""

        config = self._get_perturbation_app_config("vs_code")
        common_sections = self._build_common_perturbation_prompt_sections(
            "vs_code", config["executor"], config["focus_areas"], config["examples"]
        )

        prompt = f"""
        Decide whether to apply a VS Code perturbation during GUI task execution.

        {
            common_sections.format(
                step_idx=execution_context.step_idx,
                current_action=execution_context.current_action,
                action_history=execution_context.action_history[-3:]
                if execution_context.action_history
                else [],
                cot_context=execution_context.cot_context,
                app_states=execution_context.app_states,
                task_instruction=execution_context.task_instruction,
                target_app=scenario_spec.target_app,
                perturbation_trigger=scenario_spec.perturbation_trigger,
                available_perturbation_actions=scenario_spec.available_perturbation_actions,
                learning_objectives=scenario_spec.learning_objectives,
                target_components=scenario_spec.target_components,
                perturbation_types=[pt.value for pt in scenario_spec.perturbation_types],
            )
        }

        Return JSON object with exactly the required fields.
        """

        response = self.call_llm(prompt)
        return self._validate_perturbation_decision(response)

    def call_llm(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Call LLM to make perturbation decision"""
        response = self._call_gemini(prompt)
        self.logger.debug(f"PerturbationLLM raw response: {response[:500]}...")

        result = self.extract_json(response)
        self.logger.debug(f"PerturbationLLM extract_json result type: {type(result)}, value: {result}")

        # Validate and clean the response
        if isinstance(result, list) and len(result) > 0:
            result = result[0]  # Take the first result if it's a list
            self.logger.debug(f"PerturbationLLM took first item from list: {result}")

        if isinstance(result, dict):
            result = self._validate_perturbation_decision(result)
            self.logger.debug(f"PerturbationLLM validated result: {result}")
        else:
            self.logger.error(f"PerturbationLLM unexpected result type: {type(result)}, value: {result}")
            # Return a safe default
            result = {
                "should_apply": False,
                "reasoning": "Failed to parse LLM response",
                "api_call": "execute_python_command",
                "generated_code": "",
                "perturbation_type": "unknown",
            }

        return result

    def _validate_perturbation_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean perturbation decision format"""
        try:
            # Handle case where decision is not a dict
            if not isinstance(decision, dict):
                self.logger.error(f"Expected dict but got {type(decision)}: {decision}")
                return {
                    "should_apply": False,
                    "reasoning": "Invalid decision format",
                    "api_call": "execute_python_command",
                    "generated_code": "",
                    "perturbation_type": "unknown",
                }

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
    llm = PerturbationLLM()

    execution_context = ExecutionContext(
        step_idx=0,
        current_action='# Click on the "Data" menu to explore options for creating a Pivot Table\npyautogui.click(458, 75)',
        action_history=[],
        cot_context="",
        app_states=[
            {
                "app_type": "terminal",
                "current_view": "main_view",
                "key_elements": [],
                "task_context": "Application: gnome-shell",
                "element_count": 1897,
                "app_name": "gnome-shell",
            },
            {
                "app_type": "libreoffice_calc",
                "current_view": "main_view",
                "key_elements": [],
                "task_context": "Application: Invoices.xlsx - LibreOffice Calc",
                "element_count": 2886,
                "app_name": "Invoices.xlsx - LibreOffice Calc",
            },
        ],
        task_instruction='Create a Pivot Table in a new sheet (Sheet2) to count how many times each "Invoice No." appears.',
        task_type="libreoffice_calc",
        scenario_spec=ScenarioSpec(
            scenario_id="scenario_1",
            target_app="terminal",
            perturbation_trigger="After application starts and before the user starts creating the pivot table.",
            available_perturbation_actions="gsettings set org.gnome.desktop.interface gtk-theme 'Adwaita-dark'; gsettings set org.gnome.desktop.interface icon-theme 'Papirus-Dark'; gsettings set org.gnome.desktop.wm.preferences titlebar-font 'Sans Bold 12';",
            learning_objectives="The agent should learn to recognize terminal elements (window, text, etc.) despite changes in the GTK theme, icon theme and titlebar font.",
            target_components=["terminal window", "text input area", "window title bar", "menus"],
            perturbation_types=[PerturbationType.THEME],
        ),
    )

    scenario_spec = ScenarioSpec(
        scenario_id="scenario_1",
        target_app="terminal",
        perturbation_trigger="After application starts and before the user starts creating the pivot table.",
        available_perturbation_actions="gsettings set org.gnome.desktop.interface gtk-theme 'Adwaita-dark'; gsettings set org.gnome.desktop.interface icon-theme 'Papirus-Dark'; gsettings set org.gnome.desktop.wm.preferences titlebar-font 'Sans Bold 12';",
        learning_objectives="The agent should learn to recognize terminal elements (window, text, etc.) despite changes in the GTK theme, icon theme and titlebar font.",
        target_components=["terminal window", "text input area", "window title bar", "menus"],
        perturbation_types=[PerturbationType.THEME],
    )

    llm._decide_terminal_perturbation(execution_context, scenario_spec)

    # llm = CurriculumLLM()
    # with open("inputs.json", "r") as f:
    #     inputs = json.load(f)

    # app_states = []

    # for input in inputs:
    #     app_type = input["app_type"]
    #     seed_trajectory = input["seed_trajectory"]
    #     seed_trajectory = SeedTrajectory(**seed_trajectory)
    #     app_state = input["app_state"]
    #     curriculum_config = input["curriculum_config"]
    #     curriculum_config = CurriculumConfig(**curriculum_config)

    #     app_states.append(app_state)

    # llm.generate_scenario_specs(seed_trajectory, app_states, curriculum_config)
