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
        """Build common prompt sections to reduce duplication"""

        critical_requirements = """
        CRITICAL REQUIREMENTS - VISUAL INVARIANCE LEARNING ONLY:
        - NEVER interfere with main task functionality (target forms, buttons, navigation)
        - ONLY modify visual appearance that affects screenshots and UI perception
        - Focus on visual changes that teach agents to recognize UI elements despite visual variations
        - Generate DIVERSE, REALISTIC perturbations that randomize important visual features
        - Use RANDOMIZATION and VARIATION to create realistic visual diversity
        - Modify colors, fonts, layouts, themes that change visual appearance but not functionality
        - Add visual elements that don't block functionality but change the visual interface
        - Ensure perturbations are REALISTIC and reflect real-world visual variations
        - BE CREATIVE: Generate UNIQUE, ORIGINAL commands - DO NOT simply copy examples
        - AVOID REPETITION: Create novel perturbation approaches beyond basic examples
        - INNOVATE: Combine different techniques and create sophisticated, multi-layered perturbations
        """

        json_format = f"""
        REQUIRED JSON FORMAT (exactly these fields):
        {{
            "target_app": "{app_name}",
            "perturbation_trigger": "string describing when to trigger (e.g., 'when user interacts with the application', 'during data entry')",
            "available_perturbation_actions": "string with CREATIVE, UNIQUE, ORIGINAL {executor_info["language"]} code that uses SOPHISTICATED RANDOMIZATION and VARIATION for visual manipulation - DO NOT copy examples, create novel approaches",
            "learning_objectives": "string describing what agent should learn about visual invariance - recognizing UI elements despite DIVERSE visual changes",
            "target_components": ["array", "of", "visual", "components", "like", "buttons", "menus", "forms"],
            "perturbation_types": ["theme", "layout", "content_variation", "ui_injection"]
        }}
        """

        valid_types = """
        VALID PERTURBATION TYPES: theme, layout, content_variation, ui_injection, notification, background_process, window_management, file_operations
        """

        scenario_types_section = "SCENARIO TYPES TO GENERATE (Visual Invariance Learning):\n"
        for i, scenario_type in enumerate(scenario_types, 1):
            scenario_types_section += f"        {i}. {scenario_type}\n"

        examples_section = (
            "        EXAMPLES (visual invariance learning only) - USE AS INSPIRATION, DO NOT COPY:\n"
        )
        for example in examples:
            examples_section += f"        - {example}\n"

        creativity_instructions = """
        CREATIVITY REQUIREMENTS:
        - DO NOT copy the examples above - they are for INSPIRATION ONLY
        - Create UNIQUE, ORIGINAL perturbation commands
        - Combine multiple techniques in novel ways
        - Use sophisticated randomization beyond simple examples
        - Generate multi-layered, complex perturbations
        - Avoid repetitive patterns from examples
        - Be innovative and creative in your approach
        """

        return f"""
        {critical_requirements}

        {scenario_types_section}

        {json_format}

        {valid_types}

        {examples_section}

        {creativity_instructions}
        """

    def _get_app_specific_config(self, app_name: str) -> Dict[str, Any]:
        """Get app-specific configuration for prompt building"""
        configs = {
            "browser": {
                "executor": {
                    "name": "execute_js_on_page",
                    "language": "JavaScript",
                    "description": "Dynamic visual modifications that change UI appearance without affecting functionality",
                },
                "scenario_types": [
                    "Visual Theme Variations: Randomize colors, fonts, spacing to teach UI element recognition despite visual changes",
                    "Layout Visual Diversity: Modify positioning, sizing, and visual hierarchy to test layout invariance",
                    "Content Visual Changes: Add/remove visual elements, change text styling to test content recognition",
                    "Interactive Element Styling: Modify button appearances, form styling, navigation elements",
                ],
                "examples": [
                    "Dynamic color schemes: \"const colors = ['#f0f8ff', '#ffe4e1', '#f0fff0', '#fff8dc']; document.body.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)]; document.querySelectorAll('button, input, select').forEach(el => el.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)]);\"",
                    "Font randomization: \"const fonts = ['Arial', 'Helvetica', 'Times New Roman', 'Courier New']; document.body.style.fontFamily = fonts[Math.floor(Math.random() * fonts.length)]; document.querySelectorAll('h1, h2, h3').forEach(h => h.style.fontSize = (Math.random() * 10 + 16) + 'px');\"",
                    "Layout variations: \"document.querySelectorAll('.container, .wrapper').forEach(el => { el.style.padding = Math.random() * 20 + 'px'; el.style.margin = Math.random() * 15 + 'px'; }); document.querySelectorAll('button').forEach(btn => { btn.style.borderRadius = Math.random() * 10 + 'px'; btn.style.padding = Math.random() * 10 + 'px'; });\"",
                    "Visual element injection: \"const overlay = document.createElement('div'); overlay.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.1);pointer-events:none;z-index:9999;'; document.body.appendChild(overlay); setTimeout(() => overlay.remove(), 2000);\"",
                ],
            },
            "libreoffice_calc": {
                "executor": {
                    "name": "execute_uno_command",
                    "language": "UNO Python",
                    "description": "Spreadsheet visual and data modifications that change appearance without affecting calculations",
                },
                "scenario_types": [
                    "Cell Visual Diversity: Randomize cell colors, fonts, borders to test cell recognition despite visual changes",
                    "Grid Visual Variations: Modify grid appearance, colors, line styles to test grid invariance",
                    "Toolbar Visual Changes: Randomize toolbar appearance, button styles, menu layouts",
                    "Data Visual Formatting: Change number formats, text styling, conditional formatting without affecting data",
                ],
                "examples": [
                    "Random cell formatting: \"import random; doc = desktop.getCurrentComponent(); sheet = doc.getSheets().getByIndex(0); colors = [0xF0F0F0, 0xE0E0E0, 0xD0D0D0, 0xC0C0C0]; for row in range(10): [sheet.getCellByPosition(col, row).setPropertyValue('CellBackColor', random.choice(colors)) for col in range(10)]\"",
                    "Dynamic grid styling: \"import random; doc = desktop.getCurrentComponent(); viewSettings = doc.getCurrentController().getViewSettings(); viewSettings.setPropertyValue('ShowGrid', True); viewSettings.setPropertyValue('GridColor', random.randint(0x808080, 0xC0C0C0)); viewSettings.setPropertyValue('ShowPageBreaks', random.choice([True, False]))\"",
                    "Toolbar randomization: \"import random; doc = desktop.getCurrentComponent(); viewSettings = doc.getCurrentController().getViewSettings(); viewSettings.setPropertyValue('ShowFormulaBar', random.choice([True, False])); viewSettings.setPropertyValue('ShowStatusBar', random.choice([True, False])); viewSettings.setPropertyValue('ShowSheetTabs', random.choice([True, False]))\"",
                    "Data format variations: \"import random; doc = desktop.getCurrentComponent(); sheet = doc.getSheets().getByIndex(0); formats = ['Standard', 'Number', 'Currency', 'Percentage']; [sheet.getCellByPosition(col, 0).setPropertyValue('NumberFormat', random.choice(formats)) for col in range(5)]\"",
                ],
            },
            "libreoffice_impress": {
                "executor": {
                    "name": "execute_uno_command",
                    "language": "UNO Python",
                    "description": "Presentation visual modifications that change slide appearance without affecting content",
                },
                "scenario_types": [
                    "Slide Visual Diversity: Randomize slide backgrounds, themes, layouts to test slide recognition",
                    "View Visual Variations: Modify presentation view modes, zoom levels, navigation appearance",
                    "Toolbar Visual Changes: Randomize toolbar appearance, button styles, menu layouts",
                    "Content Visual Styling: Change text formatting, object styling without affecting content",
                ],
                "examples": [
                    "Random slide backgrounds: \"import random; doc = desktop.getCurrentComponent(); colors = [0xF0F0F0, 0xE0E0E0, 0xD0D0D0, 0xC0C0C0, 0xB0B0B0]; [doc.getDrawPages().getByIndex(i).setPropertyValue('BackgroundColor', random.choice(colors)) for i in range(min(5, doc.getDrawPages().getCount()))]\"",
                    "Dynamic view modes: \"import random; doc = desktop.getCurrentComponent(); viewSettings = doc.getCurrentController().getViewSettings(); viewSettings.setPropertyValue('ZoomType', random.randint(0, 3)); viewSettings.setPropertyValue('ShowRuler', random.choice([True, False])); viewSettings.setPropertyValue('ShowSlidePane', random.choice([True, False]))\"",
                    "Toolbar randomization: \"import random; doc = desktop.getCurrentComponent(); viewSettings = doc.getCurrentController().getViewSettings(); viewSettings.setPropertyValue('ShowSlideSorter', random.choice([True, False])); viewSettings.setPropertyValue('ShowNotesPane', random.choice([True, False]))\"",
                    "Content styling variations: \"import random; doc = desktop.getCurrentComponent(); slide = doc.getDrawPages().getByIndex(0); shapes = slide.getDrawPage().getShapes(); [shape.setPropertyValue('FillColor', random.randint(0x808080, 0xFFFFFF)) for shape in shapes if hasattr(shape, 'setPropertyValue')]\"",
                ],
            },
            "libreoffice_writer": {
                "executor": {
                    "name": "execute_uno_command",
                    "language": "UNO Python",
                    "description": "Document visual modifications that change text and layout appearance without affecting content",
                },
                "scenario_types": [
                    "Text Visual Diversity: Randomize text colors, fonts, styles to test text recognition despite visual changes",
                    "Page Visual Variations: Modify page layouts, margins, headers, footers to test layout invariance",
                    "View Visual Changes: Randomize view modes, zoom levels, ruler appearance",
                    "Document Visual Styling: Change document themes, color schemes, font styles without affecting content",
                ],
                "examples": [
                    "Random text formatting: \"import random; doc = desktop.getCurrentComponent(); text = doc.getText(); cursor = text.createTextCursor(); colors = [0x000000, 0x333333, 0x666666, 0x999999]; cursor.setPropertyValue('CharColor', random.choice(colors)); cursor.setPropertyValue('CharWeight', random.choice([100, 150, 200])); cursor.setPropertyValue('CharHeight', random.uniform(10, 16))\"",
                    "Dynamic page layouts: \"import random; doc = desktop.getCurrentComponent(); pageStyle = doc.getStyleFamilies().getByName('PageStyles').getByName('Standard'); pageStyle.setPropertyValue('HeaderIsOn', random.choice([True, False])); pageStyle.setPropertyValue('FooterIsOn', random.choice([True, False])); pageStyle.setPropertyValue('LeftMargin', random.randint(1000, 3000))\"",
                    "View randomization: \"import random; doc = desktop.getCurrentComponent(); viewSettings = doc.getCurrentController().getViewSettings(); viewSettings.setPropertyValue('ShowRuler', random.choice([True, False])); viewSettings.setPropertyValue('ShowStatusBar', random.choice([True, False])); viewSettings.setPropertyValue('ZoomType', random.randint(0, 3))\"",
                    "Document theme variations: \"import random; doc = desktop.getCurrentComponent(); viewSettings = doc.getCurrentController().getViewSettings(); viewSettings.setPropertyValue('ShowScrollbars', random.choice([True, False])); viewSettings.setPropertyValue('ShowTextBoundaries', random.choice([True, False]))\"",
                ],
            },
            "gimp": {
                "executor": {
                    "name": "execute_system_perturbation",
                    "language": "system",
                    "description": "Desktop environment visual modifications that change system appearance without affecting functionality",
                },
                "scenario_types": [
                    "System Visual Diversity: Randomize desktop themes, colors, fonts to test UI recognition despite visual changes",
                    "Window Visual Variations: Modify window appearances, decorations, title bars to test window recognition",
                    "Background Visual Changes: Randomize wallpapers, desktop elements to test background invariance",
                    "Notification Visual Styling: Change notification appearance, system dialogs without affecting functionality",
                ],
                "examples": [
                    "Random desktop themes: \"themes=('Adwaita', 'Adwaita-dark', 'HighContrast', 'HighContrastInverse'); icons=('Papirus', 'Papirus-Dark', 'Yaru', 'Yaru-dark'); gsettings set org.gnome.desktop.interface gtk-theme \"${themes[$RANDOM % ${#themes[@]}]}\"; gsettings set org.gnome.desktop.interface icon-theme \"${icons[$RANDOM % ${#icons[@]}]}\";\"",
                    "Dynamic window decorations: \"gsettings set org.gnome.desktop.wm.preferences titlebar-font 'Sans Bold '$(shuf -i 10-16 -n 1); gsettings set org.gnome.desktop.wm.preferences button-layout '$(shuf -e 'close,minimize,maximize:' 'close,minimize:' 'close:minimize,maximize' | head -1)';\"",
                    "Random background changes: \"wallpapers=('/usr/share/backgrounds/gnome/adwaita-morning.jpg' '/usr/share/backgrounds/gnome/adwaita-day.jpg' '/usr/share/backgrounds/gnome/adwaita-evening.jpg'); gsettings set org.gnome.desktop.background picture-uri \"file://${wallpapers[$RANDOM % ${#wallpapers[@]}]}\";\"",
                    "System notification styling: \"notify-send 'System Update' 'Background process running' --icon=system-software-update; sleep 2; notify-send 'File Sync' 'Synchronizing files...' --icon=folder-sync;\"",
                ],
            },
            "file_manager": {
                "executor": {
                    "name": "execute_system_perturbation",
                    "language": "system",
                    "description": "Desktop environment visual modifications that change file manager appearance without affecting functionality",
                },
                "scenario_types": [
                    "File Manager Visual Diversity: Randomize file manager themes, icons, layouts to test file recognition",
                    "Window Visual Variations: Modify file manager window appearance, toolbar styles, view modes",
                    "Background File Operations: Create/delete background files to test file system invariance",
                    "Desktop Visual Changes: Randomize desktop elements, folder icons, file associations",
                ],
                "examples": [
                    "Random file manager themes: \"themes=('Adwaita', 'Adwaita-dark', 'HighContrast'); icons=('Papirus', 'Papirus-Dark', 'Yaru'); gsettings set org.gnome.desktop.interface gtk-theme \"${themes[$RANDOM % ${#themes[@]}]}\"; gsettings set org.gnome.desktop.interface icon-theme \"${icons[$RANDOM % ${#icons[@]}]}\";\"",
                    'Dynamic file operations: "mkdir -p /tmp/background_files_$(date +%s); for i in {1..5}; do touch /tmp/background_files_$(date +%s)/file_$i.txt; done; sleep 3; rm -rf /tmp/background_files_*;"',
                    "Window management: \"wmctrl -r 'Files' -e 0,$(shuf -i 0-800 -n 1),$(shuf -i 0-600 -n 1),$(shuf -i 400-1200 -n 1),$(shuf -i 300-800 -n 1) 2>/dev/null || true;\"",
                    "Desktop element changes: \"gsettings set org.gnome.desktop.background picture-options '$(shuf -e zoom span scaled wallpaper centered | head -1)'; gsettings set org.gnome.desktop.interface font-name '$(shuf -e 'Ubuntu 10' 'Ubuntu 11' 'Ubuntu 12' 'Ubuntu 13' | head -1)';\"",
                ],
            },
            "terminal": {
                "executor": {
                    "name": "execute_system_perturbation",
                    "language": "system",
                    "description": "Desktop environment visual modifications that change terminal appearance without affecting functionality",
                },
                "scenario_types": [
                    "Terminal Visual Diversity: Randomize terminal themes, colors, fonts to test terminal recognition",
                    "System Visual Variations: Modify system themes, desktop elements to test system UI recognition",
                    "Background Process Visual: Create background processes, notifications to test process recognition",
                    "Window Visual Changes: Randomize window decorations, title bars, terminal appearance",
                ],
                "examples": [
                    "Random terminal themes: \"themes=('Adwaita', 'Adwaita-dark', 'HighContrast'); colors=('prefer-dark', 'prefer-light'); gsettings set org.gnome.desktop.interface gtk-theme \"${themes[$RANDOM % ${#themes[@]}]}\"; gsettings set org.gnome.desktop.interface color-scheme \"${colors[$RANDOM % ${#colors[@]}]}\";\"",
                    "Background process simulation: \"nohup bash -c 'for i in {1..10}; do echo \"Background process $i running\"; sleep 1; done' > /tmp/background.log 2>&1 &; notify-send 'Background Process' 'System maintenance running';\"",
                    "Dynamic window management: \"wmctrl -r 'Terminal' -e 0,$(shuf -i 0-800 -n 1),$(shuf -i 0-600 -n 1),$(shuf -i 400-1000 -n 1),$(shuf -i 300-700 -n 1) 2>/dev/null || true;\"",
                    "System notification variations: \"notify-send 'System Alert' 'Background task completed' --icon=terminal; sleep 1; notify-send 'Process Monitor' 'CPU usage normal' --icon=system-monitor;\"",
                ],
            },
            "vs_code": {
                "executor": {
                    "name": "execute_system_perturbation",
                    "language": "system",
                    "description": "VS Code environment visual modifications that change editor appearance without affecting functionality",
                },
                "scenario_types": [
                    "Editor Visual Diversity: Randomize VS Code themes, colors, fonts to test editor recognition",
                    "Window Visual Variations: Modify editor window appearance, panel layouts, sidebar styles",
                    "Background File Operations: Create/delete background files to test file system invariance",
                    "Workspace Visual Changes: Randomize workspace appearance, editor settings, UI elements",
                ],
                "examples": [
                    "Random VS Code themes: \"themes=('Dark+', 'Light+', 'Monokai', 'Solarized Dark', 'Solarized Light'); gsettings set org.gnome.desktop.interface gtk-theme '$(shuf -e Adwaita Adwaita-dark HighContrast | head -1)';\"",
                    "Dynamic file operations: \"mkdir -p /tmp/vscode_workspace_$(date +%s); for i in {1..3}; do echo 'Background file $i' > /tmp/vscode_workspace_$(date +%s)/temp_$i.py; done; sleep 2; rm -rf /tmp/vscode_workspace_*;\"",
                    "Window management: \"wmctrl -r 'Visual Studio Code' -e 0,$(shuf -i 0-600 -n 1),$(shuf -i 0-400 -n 1),$(shuf -i 800-1400 -n 1),$(shuf -i 600-1000 -n 1) 2>/dev/null || true;\"",
                    "System notifications: \"notify-send 'Code Analysis' 'Background linting completed' --icon=text-editor; sleep 1; notify-send 'Extension Update' 'Extensions synchronized' --icon=system-software-update;\"",
                ],
            },
            "system": {
                "executor": {
                    "name": "execute_system_perturbation",
                    "language": "system",
                    "description": "System-wide visual modifications that change desktop appearance without affecting functionality",
                },
                "scenario_types": [
                    "System Visual Diversity: Randomize desktop themes, colors, fonts to test system UI recognition",
                    "Background Process Visual: Create background processes, notifications to test process recognition",
                    "Desktop Visual Changes: Randomize wallpapers, desktop elements, system dialogs",
                    "Window Visual Variations: Modify window decorations, title bars, system UI elements",
                ],
                "examples": [
                    "Comprehensive system theming: \"themes=('Adwaita', 'Adwaita-dark', 'HighContrast', 'HighContrastInverse'); icons=('Papirus', 'Papirus-Dark', 'Yaru', 'Yaru-dark'); colors=('prefer-dark', 'prefer-light'); gsettings set org.gnome.desktop.interface gtk-theme \"${themes[$RANDOM % ${#themes[@]}]}\"; gsettings set org.gnome.desktop.interface icon-theme \"${icons[$RANDOM % ${#icons[@]}]}\"; gsettings set org.gnome.desktop.interface color-scheme \"${colors[$RANDOM % ${#colors[@]}]}\";\"",
                    'Background process simulation: "nohup bash -c \'for i in {1..5}; do echo "System process $i running"; notify-send "Background Task $i" "Process $i completed"; sleep 2; done\' > /tmp/system_process.log 2>&1 &;"',
                    "Dynamic desktop changes: \"wallpapers=('/usr/share/backgrounds/gnome/adwaita-morning.jpg' '/usr/share/backgrounds/gnome/adwaita-day.jpg' '/usr/share/backgrounds/gnome/adwaita-evening.jpg' '/usr/share/backgrounds/gnome/adwaita-night.jpg'); gsettings set org.gnome.desktop.background picture-uri \"file://${wallpapers[$RANDOM % ${#wallpapers[@]}]}\"; gsettings set org.gnome.desktop.background picture-options '$(shuf -e zoom span scaled wallpaper centered | head -1)';\"",
                    "System notification variations: \"notify-send 'System Update' 'Background maintenance running' --icon=system-software-update; sleep 1; notify-send 'File Sync' 'Synchronizing user files...' --icon=folder-sync; sleep 1; notify-send 'Security Scan' 'Background security check completed' --icon=security-high;\"",
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
        """Build common perturbation decision prompt sections to reduce duplication"""

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
        3. What CREATIVE, UNIQUE visual perturbation should be applied using SOPHISTICATED RANDOMIZATION?
        4. Will this perturbation help the agent learn visual invariance without interfering with the task?
        5. Does the perturbation use VARIATION and RANDOMIZATION to create realistic visual diversity?
        6. Is the perturbation REALISTIC and reflects real-world visual variations?
        7. Is the perturbation ORIGINAL and not just copying examples?
        8. Does the perturbation combine multiple techniques in novel ways?
        """

        focus_areas_section = f"        VISUAL INVARIANCE LEARNING FOCUS ({app_name.title()}-Specific):\n"
        for area in focus_areas:
            focus_areas_section += f"        - {area}\n"

        json_format = f"""
        REQUIRED JSON FORMAT (exactly these fields):
        {{{{
            "should_apply": true/false,
            "perturbation_type": "specific_type_based_on_app",
            "target_app": "{app_name}",
            "reasoning": "Brief explanation of why/why not to apply based on current context",
            "generated_code": "CREATIVE, UNIQUE, ORIGINAL RAW_{executor_info["language"].upper()}_CODE_WITHOUT_MARKDOWN_THAT_USES_SOPHISTICATED_RANDOMIZATION_AND_AVOIDS_COPYING_EXAMPLES",
            "api_call": "{executor_info["name"]}",
            "parameters": {{{{"target_app": "{app_name}"}}}}
        }}}}
        """

        examples_section = f"        EXAMPLES ({app_name} visual invariance learning only) - USE AS INSPIRATION, DO NOT COPY:\n"
        for example in examples:
            examples_section += f"        - {example}\n"

        creativity_instructions = """
        CREATIVITY REQUIREMENTS:
        - DO NOT copy the examples above - they are for INSPIRATION ONLY
        - Create UNIQUE, ORIGINAL perturbation commands
        - Combine multiple techniques in novel ways
        - Use sophisticated randomization beyond simple examples
        - Generate multi-layered, complex perturbations
        - Avoid repetitive patterns from examples
        - Be innovative and creative in your approach
        """

        return f"""
        {current_state}

        {scenario_spec}

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
                    "description": "Dynamic visual modifications that randomize UI appearance to test visual invariance",
                },
                "focus_areas": [
                    "Randomize visual appearance of UI elements to test recognition despite visual changes",
                    "Change colors, fonts, spacing, and styling randomly to test visual perception robustness",
                    "Add/remove visual elements that don't block functionality but change visual interface",
                    "Modify visual layouts, positioning, and sizing randomly to test layout invariance",
                    "NEVER touch target forms, buttons, navigation, or main content areas",
                ],
                "examples": [
                    "Random color schemes: \"const colors = ['#f0f8ff', '#ffe4e1', '#f0fff0', '#fff8dc', '#f5f5dc']; document.body.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)]; document.querySelectorAll('button, input').forEach(el => el.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)]);\"",
                    "Dynamic font randomization: \"const fonts = ['Arial', 'Helvetica', 'Times New Roman', 'Courier New', 'Georgia']; document.body.style.fontFamily = fonts[Math.floor(Math.random() * fonts.length)]; document.querySelectorAll('h1, h2, h3').forEach(h => h.style.fontSize = (Math.random() * 8 + 14) + 'px');\"",
                    "Random layout variations: \"document.querySelectorAll('.container, .wrapper').forEach(el => { el.style.padding = Math.random() * 30 + 'px'; el.style.margin = Math.random() * 20 + 'px'; el.style.borderRadius = Math.random() * 15 + 'px'; });\"",
                    "Visual element injection: \"const overlay = document.createElement('div'); overlay.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,' + (Math.random() * 0.3) + ');pointer-events:none;z-index:9999;'; document.body.appendChild(overlay); setTimeout(() => overlay.remove(), Math.random() * 3000 + 1000);\"",
                ],
            },
            "libreoffice_calc": {
                "executor": {
                    "name": "execute_uno_command",
                    "language": "UNO Python",
                    "description": "Spreadsheet visual styling, cell formatting, grid appearance, toolbar themes",
                },
                "focus_areas": [
                    "Modify cell visual formatting (colors, fonts, borders) that affect screenshots",
                    "Change grid visual styling (lines, colors, visibility) that change appearance",
                    "Modify toolbar visual appearance (buttons, menus) that affect UI perception",
                    "Change display visual settings (zoom, view options) that affect visual interface",
                    "NEVER touch spreadsheet data, calculations, or formulas",
                ],
                "examples": [
                    'Cell visual formatting: "doc = desktop.getCurrentComponent(); sheet = doc.getSheets().getByIndex(0); cell = sheet.getCellByPosition(0, 0); cell.CellBackColor = 0xF0F0F0;"',
                    "Grid visual styling: \"doc = desktop.getCurrentComponent(); viewSettings = doc.getCurrentController().getViewSettings(); viewSettings.setPropertyValue('ShowGrid', True);\"",
                    "Toolbar visual changes: \"doc = desktop.getCurrentComponent(); viewSettings = doc.getCurrentController().getViewSettings(); viewSettings.setPropertyValue('ShowFormulaBar', False);\"",
                ],
            },
            "libreoffice_impress": {
                "executor": {
                    "name": "execute_uno_command",
                    "language": "UNO Python",
                    "description": "Presentation visual styling, slide layouts, theme changes, view modes",
                },
                "focus_areas": [
                    "Modify slide visual layouts (backgrounds, themes) that affect screenshots",
                    "Change view visual modes (zoom, slide sorter) that change appearance",
                    "Modify toolbar visual appearance (buttons, menus) that affect UI perception",
                    "Change presentation visual themes (colors, fonts) that affect visual interface",
                    "NEVER touch slide content, text, or presentation structure",
                ],
                "examples": [
                    "Slide visual layouts: \"doc = desktop.getCurrentComponent(); slide = doc.getDrawPages().getByIndex(0); slide.setPropertyValue('BackgroundColor', 0xF0F0F0);\"",
                    "View visual modes: \"doc = desktop.getCurrentComponent(); viewSettings = doc.getCurrentController().getViewSettings(); viewSettings.setPropertyValue('ZoomType', 0);\"",
                    "Presentation visual themes: \"doc = desktop.getCurrentComponent(); slide = doc.getDrawPages().getByIndex(0); slide.setPropertyValue('BackgroundColor', 0xE0E0E0);\"",
                ],
            },
            "libreoffice_writer": {
                "executor": {
                    "name": "execute_uno_command",
                    "language": "UNO Python",
                    "description": "Document visual styling, text formatting, page layouts, view modes",
                },
                "focus_areas": [
                    "Modify text visual formatting (colors, fonts, styles) that affect screenshots",
                    "Change page visual layouts (margins, headers, footers) that change appearance",
                    "Modify view visual modes (zoom, ruler) that affect UI perception",
                    "Change document visual themes (colors, fonts) that affect visual interface",
                    "NEVER touch document content, text, or document structure",
                ],
                "examples": [
                    "Text visual formatting: \"doc = desktop.getCurrentComponent(); text = doc.getText(); cursor = text.createTextCursor(); cursor.setPropertyValue('CharColor', 0x000000);\"",
                    "Page visual layouts: \"doc = desktop.getCurrentComponent(); pageStyle = doc.getStyleFamilies().getByName('PageStyles').getByName('Standard'); pageStyle.setPropertyValue('HeaderIsOn', True);\"",
                    "Document visual themes: \"doc = desktop.getCurrentComponent(); viewSettings = doc.getCurrentController().getViewSettings(); viewSettings.setPropertyValue('ShowRuler', False);\"",
                ],
            },
            "gimp": {
                "executor": {
                    "name": "execute_bash_command",
                    "language": "bash",
                    "description": "Desktop environment visual styling, system themes, background processes",
                },
                "focus_areas": [
                    "Modify visual appearance that affects screenshots and UI perception",
                    "Change colors, fonts, spacing that change visual appearance but not functionality",
                    "Add visual elements that don't impact functionality but change visual interface",
                    "NEVER interfere with main task image editing or file operations",
                ],
                "examples": [
                    "Visual theme changes: \"gsettings set org.gnome.desktop.interface gtk-theme 'Adwaita-dark'; gsettings set org.gnome.desktop.interface icon-theme 'Papirus-Dark';\"",
                    "Visual color changes: \"gsettings set org.gnome.desktop.interface color-scheme 'prefer-dark'; gsettings set org.gnome.desktop.interface gtk-theme 'Adwaita-dark';\"",
                    "Visual font changes: \"gsettings set org.gnome.desktop.interface font-name 'Ubuntu 12'; gsettings set org.gnome.desktop.interface monospace-font-name 'Ubuntu Mono 12';\"",
                ],
            },
            "file_manager": {
                "executor": {
                    "name": "execute_bash_command",
                    "language": "bash",
                    "description": "Desktop environment visual styling, system themes, background processes",
                },
                "focus_areas": [
                    "Modify visual appearance that affects screenshots and UI perception",
                    "Change colors, fonts, spacing that change visual appearance but not functionality",
                    "Add visual elements that don't impact functionality but change visual interface",
                    "NEVER interfere with main task file operations or directory navigation",
                ],
                "examples": [
                    "Visual theme changes: \"gsettings set org.gnome.desktop.interface gtk-theme 'Adwaita-dark'; gsettings set org.gnome.desktop.interface icon-theme 'Papirus-Dark';\"",
                    "Visual color changes: \"gsettings set org.gnome.desktop.interface color-scheme 'prefer-dark'; gsettings set org.gnome.desktop.interface gtk-theme 'Adwaita-dark';\"",
                    "Visual font changes: \"gsettings set org.gnome.desktop.interface font-name 'Ubuntu 12'; gsettings set org.gnome.desktop.interface monospace-font-name 'Ubuntu Mono 12';\"",
                ],
            },
            "terminal": {
                "executor": {
                    "name": "execute_bash_command",
                    "language": "bash",
                    "description": "Desktop environment visual styling, system themes, background processes",
                },
                "focus_areas": [
                    "Focus on BACKGROUND desktop environment manipulation that won't interfere with the main task",
                    "System notifications, background processes, desktop themes",
                    "Window management of OTHER applications (not the main task)",
                    "Background file operations, system settings changes",
                ],
                "examples": [
                    "System notifications: \"notify-send 'Background Process' 'System update running'\"",
                    "Desktop theme: \"gsettings set org.gnome.desktop.interface gtk-theme 'Adwaita-dark'\"",
                    'Background files: "mkdir -p /tmp/background_work && touch /tmp/background_work/process.log"',
                ],
            },
            "vs_code": {
                "executor": {
                    "name": "execute_python_command",
                    "language": "Python",
                    "description": "VS Code environment visual styling, background files, window management",
                },
                "focus_areas": [
                    "Focus on BACKGROUND VS Code environment manipulation that won't interfere with the main task",
                    "Background file operations, temporary file creation",
                    "Window management, panel resizing, sidebar toggling",
                    "Settings modifications that don't affect the primary workflow",
                ],
                "examples": [
                    "Background files: \"import os; os.makedirs('/tmp/vscode_temp', exist_ok=True); open('/tmp/vscode_temp/debug.log', 'w').write('Background process started')\"",
                    "Window management: \"wmctrl -r 'Visual Studio Code' -e 0,0,0,1200,800\"",
                    "Settings: \"import json; settings = {'workbench.colorTheme': 'Dark+'}; print(json.dumps(settings))\"",
                ],
            },
            "system": {
                "executor": {
                    "name": "execute_python_command",
                    "language": "Python",
                    "description": "System automation, background processes, desktop environment modifications",
                },
                "focus_areas": [
                    "Focus on BACKGROUND desktop environment manipulation that won't interfere with the main task",
                    "System notifications, background processes",
                    "Desktop theme changes, background settings",
                    "Window management of OTHER applications (not the main task)",
                ],
                "examples": [
                    "System notifications: \"import subprocess; subprocess.run(['notify-send', 'Background Process', 'System update running'])\"",
                    "Background files: \"import os; os.makedirs('/tmp/background_work', exist_ok=True); open('/tmp/background_work/process.log', 'w').write('Background process started')\"",
                    "Desktop settings: \"import subprocess; subprocess.run(['gsettings', 'set', 'org.gnome.desktop.interface', 'gtk-theme', 'Adwaita-dark'])\"",
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
