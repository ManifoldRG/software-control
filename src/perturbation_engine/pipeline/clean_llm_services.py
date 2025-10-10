"""
Clean LLM Services: Simplified interfaces for LLM interactions with comprehensive operation awareness
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types

from perturbation_engine.pipeline.app_state_utils import (
    normalize_ui_elements,
    normalize_window_states,
)
from perturbation_engine.pipeline.data_models import (
    ApiCallType,
    CurriculumConfig,
    ExecutionContext,
    GeneratedTrajectory,
    PerturbationCategory,
    PerturbationIntensity,
    PerturbationType,
    ScenarioSpec,
    SeedTrajectory,
    UIElement,
    WindowState,
)

PROMPT_CONSTANTS = {
    "task_analysis_role": "You are an expert in software automation and curriculum design. Analyze this task to understand its characteristics, complexity, domain, and perturbation opportunities.",
    "curriculum_role": "You are an expert in curriculum design for software control systems. Generate EXACTLY {scenario_count} diverse, strategic perturbation scenarios that teach visual invariance and robustness.",
    "perturbation_role": "You are an expert in GUI automation perturbation. Decide whether to apply perturbation at the current step, considering procedural memory context and available operations.",
    "complexity_options": "simple|moderate|complex",
    "domain_options": "office|web|multimedia|development|system|general",
    "timing_options": "initial|runtime|between_steps",
    "target_scope_options": "system|app|file|content",
    "educational_value_options": "high|medium|low",
    "risk_level_options": "low|medium|high",
    "workflow_type_options": "sequential|parallel|iterative|conditional",
    "analysis_requirements": [
        "Assess task complexity (simple/moderate/complex) based on:",
        "  - Number of steps and requirements required",
        "  - Cognitive load",
        "  - Error potential",
        "  - Dependencies between actions",
        "",
        "Identify task domain (office/web/multimedia/development/system/general) based on:",
        "  - Primary applications involved",
        "  - Task objectives",
        "  - User workflows",
        "",
        "Derive learning objectives for visual invariance training:",
        "  - What visual changes would challenge this task?",
        "  - What UI elements are critical for success?",
        "  - What perturbation types would be most educational?",
        "",
        "Identify perturbation opportunities with focus on:",
        "  - System-level randomization (themes, wallpapers, desktop layout)",
        "  - Content/data randomization (file contents, media properties)",
        "  - App-specific randomization (settings, configurations)",
        "  - Cross-app interference (background processes, notifications)",
        "",
        "Focus on perturbation feasibility:",
        "  - Which elements must remain accessible for the task?",
        "  - What timing constraints exist?",
    ],
    "good_analysis_examples": [
        'VLC task: "Focus on system themes and content randomization since VLC has limited GUI manipulation"',
        'Chrome task: "Rich GUI manipulation available - can use CSS injection, DOM modification, and theme changes"',
        'Office task: "Content randomization through file modifications and system-level theme changes"',
    ],
    "diversity_requirements": [
        "Cover ALL perturbation categories: {categories}",
        "Use different perturbation types: {types}",
        "Mix intensities: {intensities}",
        "Include both initial_state_perturbation and runtime_perturbation",
        "Vary target_scopes: system, app, file, content",
        "Different timings: initial, runtime, between_steps",
    ],
    "perturbation_criteria": [
        "Does the current step match the perturbation trigger?",
        "Would perturbation enhance learning without breaking the task?",
        "Is the perturbation coherent with previous perturbations?",
        "Does the target app have the necessary elements for perturbation?",
        "Would the perturbation maintain task feasibility?",
        "Will the perturbation make the next action target element unreachable?",
    ],
    "decision_examples": [
        'Trigger: "When VLC is launched" + Current: "VLC startup" → should_apply: true',
        'Trigger: "During video playback" + Current: "Clicking play button" → should_apply: true',
        'Trigger: "When interacting with VLC" + Current: "Chrome navigation" → should_apply: false',
        'Previous perturbations: 3 recent + Current: "Minor action" → should_apply: false (avoid over-perturbation)',
    ],
    "command_examples": [
        "System theme: \"change_wallpaper('/path/to/dark_wallpaper.jpg')\"",
        "VLC settings: \"VLCTools.set_settings('qt-theme', 'dark')\"",
        "CSS injection: \"inject_css('button {{ background-color: #ff0000; }}')\"",
        "Content modification: \"modify_file_content('/path/to/file', 'new_content')\"",
    ],
    "formatting_templates": {
        "opportunity_format": "{i}. {element_type} - {perturbation_type} (timing: {timing}, intensity: {intensity}, educational_value: {educational_value}, risk: {risk_level})",
        "risk_format": "{i}. {element_type}: {risk_description} (mitigation: {mitigation})",
        "characteristic_format": "- {key}: {value}",
    },
}


class ProceduralMemory:
    """Simple procedural memory for perturbation coherence"""

    def __init__(self):
        self.perturbation_history = []
        self.app_contexts = {}

    def add_perturbation(
        self, step_idx: int, target_app: str, perturbation_type: str, command: str, app_state: Dict[str, Any]
    ):
        """Add perturbation to memory"""
        self.perturbation_history.append(
            {
                "step_idx": step_idx,
                "target_app": target_app,
                "perturbation_type": perturbation_type,
                "command": command,
                "app_state": app_state,
                "timestamp": step_idx,  # Using step as timestamp
            }
        )

        # Update app context
        if target_app not in self.app_contexts:
            self.app_contexts[target_app] = []
        self.app_contexts[target_app].append(
            {"step_idx": step_idx, "perturbation_type": perturbation_type, "command": command}
        )

    def get_recent_perturbations(self, target_app: str = None, limit: int = 3) -> List[Dict[str, Any]]:
        """Get recent perturbations for coherence"""
        if target_app:
            app_perturbations = [p for p in self.perturbation_history if p["target_app"] == target_app]
            return app_perturbations[-limit:] if app_perturbations else []
        return self.perturbation_history[-limit:] if self.perturbation_history else []

    def get_coherence_context(self, target_app: str) -> str:
        """Get coherence context for LLM"""
        recent = self.get_recent_perturbations(target_app, 2)
        if not recent:
            return f"No recent perturbations for {target_app}"

        context_parts = []
        for p in recent:
            context_parts.append(f"Step {p['step_idx']}: {p['perturbation_type']} - {p['command'][:50]}...")

        return f"Recent perturbations for {target_app}:\n" + "\n".join(context_parts)


class OperationCatalog:
    """Comprehensive catalog of ALL available operations"""

    def __init__(self):
        self.catalog = self._build_comprehensive_catalog()

    def _build_comprehensive_catalog(self) -> Dict[str, Any]:
        """Build comprehensive catalog of ALL available operations"""
        return {
            "app_tools": {
                "vlc": self._load_vlc_operations(),
                "chrome": self._load_chrome_operations(),
                "google_chrome": self._load_chrome_operations(),
                "code": self._load_code_operations(),
                "libreoffice_calc": self._load_calc_operations(),
                "libreoffice_writer": self._load_writer_operations(),
                "libreoffice_impress": self._load_impress_operations(),
            },
            "system_operations": self._load_system_operations(),
            "server_endpoints": self._load_server_endpoints(),
            "python_controller": self._load_python_controller_operations(),
            "visual_manipulation": self._load_visual_manipulation_operations(),
            "freeform_operations": self._load_freeform_operations(),
            "perturbation_categories": [
                "theme",
                "layout",
                "content_variation",
                "ui_state",
                "accessibility",
                "performance",
                "network",
                "file_system",
                "playback",
                "settings",
                "playlist",
                "navigation",
                "tabs",
                "bookmarks",
                "devtools",
                "file_ops",
                "editing",
                "window_management",
                "visual_randomization",
                "gui_manipulation",
                "css_injection",
                "dom_modification",
            ],
        }

    def _load_vlc_operations(self) -> Dict[str, List[str]]:
        """Load VLC-specific operations from tools/apis/vlc.json"""
        return {
            "playback": ["play", "pause", "stop", "next", "previous", "seek", "set_volume"],
            "settings": ["set_settings", "get_settings", "set_fullscreen", "set_theme"],
            "playlist": ["add_to_playlist", "remove_from_playlist", "clear_playlist", "get_playlist"],
            "ui": ["set_layout", "toggle_controls", "set_window_size", "set_zoom"],
            "media": ["load_media", "get_media_info", "set_audio_track", "set_subtitle_track"],
        }

    def _load_chrome_operations(self) -> Dict[str, List[str]]:
        """Load Chrome-specific operations from tools/apis/google_chrome.json"""
        return {
            "navigation": ["navigate", "go_back", "go_forward", "refresh", "reload"],
            "tabs": ["open_tab", "close_tab", "switch_tab", "new_tab", "get_tabs"],
            "bookmarks": ["bookmark_page", "get_bookmarks", "delete_bookmark", "create_bookmark_folder"],
            "settings": ["open_settings", "set_theme", "set_language", "set_homepage"],
            "devtools": ["open_devtools", "inspect_element", "console_log", "network_monitor"],
            "extensions": ["install_extension", "disable_extension", "get_extensions"],
            "security": ["clear_cookies", "clear_cache", "set_privacy_settings"],
        }

    def _load_code_operations(self) -> Dict[str, List[str]]:
        """Load Code editor operations from tools/apis/code.json"""
        return {
            "file_ops": ["open_file", "save_file", "create_file", "delete_file", "rename_file"],
            "editing": ["insert_text", "delete_text", "find_replace", "format_code", "comment_code"],
            "navigation": ["go_to_line", "find_symbol", "go_to_definition", "find_references"],
            "settings": ["set_theme", "set_font_size", "toggle_sidebar", "set_indentation"],
            "git": ["git_status", "git_commit", "git_push", "git_pull", "git_branch"],
            "debugging": ["set_breakpoint", "start_debugging", "step_over", "step_into"],
        }

    def _load_calc_operations(self) -> Dict[str, List[str]]:
        """Load LibreOffice Calc operations from tools/apis/libreoffice_calc.json"""
        return {
            "workbook": ["get_workbook_info", "save", "export_to_csv", "export_to_pdf"],
            "sheets": ["switch_active_sheet", "rename_sheet", "copy_sheet", "reorder_sheets"],
            "data": ["get_column_data", "set_column_values", "set_cell_value", "sort_column"],
            "formatting": ["format_range", "highlight_range", "set_number_format", "merge_cells"],
            "charts": ["create_chart", "set_chart_legend_position"],
            "layout": ["freeze_panes", "adjust_column_width", "adjust_row_height", "set_zoom_level"],
            "analysis": ["create_pivot_table", "set_validation_list", "hide_row_data", "reorder_columns"],
            "transforms": ["transpose_range"],
        }

    def _load_writer_operations(self) -> Dict[str, List[str]]:
        """Load LibreOffice Writer operations from tools/apis/libreoffice_writer.json"""
        return {
            "document": ["save", "export_to_pdf"],
            "text": ["write_text", "find_and_replace", "change_text_case", "capitalize_words"],
            "formatting": ["set_color", "set_font", "set_font_size", "set_line_spacing", "set_strikethrough"],
            "layout": ["set_paragraph_alignment", "insert_page_break", "add_page_numbers"],
            "content": ["insert_formula_at_cursor", "insert_image_at_cursor"],
            "styling": ["remove_highlighting", "find_highlighted_text", "set_default_font"],
        }

    def _load_impress_operations(self) -> Dict[str, List[str]]:
        """Load LibreOffice Impress operations from tools/apis/libreoffice_impress.json"""
        return {
            "presentation": ["save", "save_as", "export_to_image"],
            "slides": ["go_to_slide", "get_slide_count", "duplicate_slide", "set_slide_orientation"],
            "content": ["write_text", "insert_image", "insert_file", "delete_content"],
            "formatting": ["set_style", "set_background_color", "set_text_color", "set_text_strikethrough"],
            "layout": ["position_box", "set_textbox_alignment", "set_slide_background"],
            "settings": [
                "configure_auto_save",
                "configure_display_settings",
                "set_slide_font",
                "set_slide_number_color",
            ],
        }

    def _load_system_operations(self) -> Dict[str, List[str]]:
        """Load system-level operations from SetupController"""
        return {
            "file_system": ["get_file", "upload_file", "list_directory", "get_desktop_path", "download_file"],
            "system_info": [
                "get_screenshot",
                "get_accessibility_tree",
                "get_terminal_output",
                "get_screen_size",
                "get_window_size",
                "get_platform",
            ],
            "execution": [
                "execute_command",
                "execute_with_verification",
                "run_python_script",
                "run_bash_script",
            ],
            "window_management": ["activate_window", "close_window", "launch_app", "open_file"],
            "system_settings": ["change_wallpaper", "get_wallpaper", "get_cursor_position"],
            "recording": ["start_recording", "end_recording"],
        }

    def _load_server_endpoints(self) -> Dict[str, List[str]]:
        """Load all available server endpoints from main.py"""
        return {
            "setup": [
                "/setup/execute",
                "/setup/execute_with_verification",
                "/setup/launch",
                "/setup/upload",
                "/setup/change_wallpaper",
                "/setup/download_file",
                "/setup/open_file",
                "/setup/activate_window",
                "/setup/close_window",
            ],
            "core": ["/execute", "/execute_with_verification", "/screenshot", "/accessibility", "/terminal"],
            "system": [
                "/screen_size",
                "/window_size",
                "/desktop_path",
                "/wallpaper",
                "/list_directory",
                "/file",
                "/platform",
                "/cursor_position",
            ],
            "execution": ["/run_python", "/run_bash_script"],
            "recording": ["/start_recording", "/end_recording"],
        }

    def _load_python_controller_operations(self) -> Dict[str, List[str]]:
        """Load PythonController operations"""
        return {
            "screenshots": ["get_screenshot"],
            "accessibility": ["get_accessibility_tree"],
            "terminal": ["get_terminal_output"],
            "files": ["get_file"],
            "execution": ["execute_python_command", "run_python_script", "run_bash_script"],
            "actions": ["execute_action"],
            "recording": ["start_recording", "end_recording"],
            "system_info": [
                "get_vm_platform",
                "get_vm_screen_size",
                "get_vm_window_size",
                "get_vm_wallpaper",
                "get_vm_desktop_path",
                "get_vm_directory_tree",
            ],
        }

    def _load_visual_manipulation_operations(self) -> Dict[str, List[str]]:
        """Load visual manipulation operations for GUI randomization"""
        return {
            "css_injection": [
                "inject_css",
                "modify_element_style",
                "change_color_scheme",
                "randomize_fonts",
                "alter_spacing",
                "modify_borders",
                "change_shadows",
                "adjust_opacity",
                "modify_transitions",
                "change_backgrounds",
            ],
            "dom_modification": [
                "add_fake_elements",
                "modify_element_text",
                "change_element_attributes",
                "reorder_elements",
                "hide_show_elements",
                "duplicate_elements",
                "modify_element_classes",
                "change_element_ids",
                "alter_element_hierarchy",
            ],
            "theme_randomization": [
                "randomize_color_palette",
                "change_theme_variant",
                "modify_accent_colors",
                "alter_contrast_levels",
                "change_brightness",
                "modify_saturation",
                "randomize_gradients",
                "change_icon_styles",
                "alter_button_styles",
            ],
            "layout_perturbation": [
                "randomize_element_positions",
                "modify_element_sizes",
                "change_alignment",
                "alter_margins_padding",
                "modify_grid_layouts",
                "change_flex_properties",
                "randomize_z_index",
                "modify_overflow_settings",
                "change_display_properties",
            ],
            "typography_randomization": [
                "randomize_font_families",
                "change_font_sizes",
                "modify_font_weights",
                "alter_line_heights",
                "change_letter_spacing",
                "modify_text_decoration",
                "randomize_text_colors",
                "change_text_shadows",
                "alter_text_transforms",
            ],
            "animation_effects": [
                "add_random_animations",
                "modify_transition_durations",
                "change_easing_functions",
                "randomize_keyframes",
                "alter_animation_delays",
                "modify_transform_effects",
                "change_animation_directions",
                "add_hover_effects",
                "modify_scroll_behavior",
            ],
            "accessibility_perturbation": [
                "modify_aria_labels",
                "change_tab_order",
                "alter_focus_styles",
                "modify_screen_reader_text",
                "change_contrast_ratios",
                "alter_text_scaling",
                "modify_keyboard_navigation",
                "change_high_contrast_mode",
                "alter_color_blind_support",
            ],
        }

    def _load_freeform_operations(self) -> Dict[str, List[str]]:
        """Load freeform operations for creative GUI manipulation"""
        return {
            "python_execution": [
                "execute_python_code",
                "run_custom_script",
                "inject_python_module",
                "modify_runtime_behavior",
                "execute_dynamic_code",
                "run_conditional_logic",
                "execute_async_operations",
                "run_background_tasks",
                "execute_ui_automation",
            ],
            "javascript_injection": [
                "inject_javascript",
                "modify_page_behavior",
                "execute_dom_scripts",
                "run_custom_functions",
                "modify_event_handlers",
                "execute_async_js",
                "inject_jquery_code",
                "run_vanilla_js",
                "execute_framework_code",
            ],
            "bash_automation": [
                "execute_bash_commands",
                "run_shell_scripts",
                "modify_system_files",
                "execute_file_operations",
                "run_network_commands",
                "execute_process_management",
                "run_package_management",
                "execute_system_configuration",
                "run_custom_automation",
            ],
            "playwright_automation": [
                "execute_playwright_actions",
                "modify_page_content",
                "inject_custom_elements",
                "execute_mouse_actions",
                "run_keyboard_automation",
                "modify_page_navigation",
                "execute_screenshot_operations",
                "run_element_interactions",
                "execute_custom_workflows",
            ],
            "file_system_manipulation": [
                "modify_config_files",
                "create_temp_files",
                "alter_user_preferences",
                "modify_theme_files",
                "change_icon_files",
                "alter_font_files",
                "modify_css_files",
                "change_javascript_files",
                "alter_resource_files",
            ],
            "network_perturbation": [
                "modify_network_requests",
                "inject_custom_responses",
                "alter_api_responses",
                "modify_http_headers",
                "change_request_timeouts",
                "alter_connection_settings",
                "inject_network_delays",
                "modify_dns_responses",
                "alter_ssl_settings",
            ],
            "system_integration": [
                "modify_system_settings",
                "change_desktop_environment",
                "alter_window_manager",
                "modify_display_settings",
                "change_input_methods",
                "alter_system_sounds",
                "modify_notification_settings",
                "change_power_management",
                "alter_security_settings",
            ],
        }

    def get_operations_for_app(self, app_name: str) -> Dict[str, List[str]]:
        """Get operations for specific app"""
        return self.catalog["app_tools"].get(app_name.lower(), {})

    def format_operations_for_llm(self, window_states: List[Any] = None) -> str:
        """Format operations for LLM prompt based on app states"""
        if not window_states:
            self.logger.error("No window states provided to format_operations_for_llm")
            raise ValueError("No window states provided")

        normalized_states = normalize_window_states(window_states)
        formatted_operations = []

        for window_state in normalized_states:
            app_name = window_state.app_name
            app_ops = self.get_operations_for_app(app_name)

            if app_ops:
                app_formatted = f"OPERATIONS FOR {app_name.upper()}:\n"
                for category, operations in app_ops.items():
                    app_formatted += f"  {category}: {', '.join(operations)}\n"
                formatted_operations.append(app_formatted)
            else:
                # Include system operations if no app-specific operations found
                formatted_operations.append(
                    f"OPERATIONS FOR {app_name.upper()}:\n  system: Using system-level operations\n"
                )

        # Always include system operations as fallback
        system_ops = self.catalog["system_operations"]
        system_formatted = "SYSTEM OPERATIONS (available for all apps):\n"
        for category, operations in system_ops.items():
            system_formatted += f"  {category}: {', '.join(operations)}\n"
        formatted_operations.append(system_formatted)

        # Include visual manipulation operations for GUI randomization
        visual_ops = self.catalog["visual_manipulation"]
        visual_formatted = "VISUAL MANIPULATION OPERATIONS (for GUI randomization):\n"
        for category, operations in visual_ops.items():
            visual_formatted += f"  {category}: {', '.join(operations)}\n"
        formatted_operations.append(visual_formatted)

        # Include freeform operations for creative manipulation
        freeform_ops = self.catalog["freeform_operations"]
        freeform_formatted = "FREEFORM OPERATIONS (for creative GUI manipulation):\n"
        for category, operations in freeform_ops.items():
            freeform_formatted += f"  {category}: {', '.join(operations)}\n"
        formatted_operations.append(freeform_formatted)

        return "\n".join(formatted_operations)


class CleanLLM:
    """Clean, simplified LLM interface"""

    def __init__(self, model_name: str = "gemini-2.5-flash-lite"):
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client = None

        if self.api_key:
            self.client = genai.Client()
        else:
            self.logger.warning("Gemini API not available - using mock responses")

    def call_llm(self, prompt: str) -> str:
        """Call LLM with prompt"""
        if not self.client:
            return '{"error": "Mock response - API not available"}'

        retries = 0
        max_retries = 3
        while retries < max_retries:
            try:
                retries += 1
                if self.model_name.startswith("gemini-"):
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            thinking_config=types.ThinkingConfig(thinking_budget=0)
                        ),
                    )
                    return response.text
                elif self.model_name.startswith("openrouter"):
                    from openai import OpenAI

                    client = OpenAI(
                        base_url="https://openrouter.ai/api/v1",
                        api_key=os.getenv("OPENROUTER_API_KEY"),
                    )

                    completion = client.chat.completions.create(
                        extra_headers={},
                        extra_body={},
                        model="moonshotai/kimi-dev-72b:free",
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return completion.choices[0].message.content
            except Exception as e:
                self.logger.error(f"Error calling LLM: {e}, retrying {retries}/{max_retries}...")

        return f"{'error: LLM call failed after {max_retries} attempts'}"

    def extract_json(self, response: str) -> List[Dict[str, Any]]:
        """Extract JSON from LLM response"""
        try:
            # Simple JSON extraction
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response.strip()

            parsed = json.loads(json_str)
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict):
                return [parsed]
            return []

        except Exception as e:
            self.logger.error(f"Error extracting JSON: {e}")
            return []


class CleanCurriculumGenerator(CleanLLM):
    """LLM-driven curriculum generator for diverse and strategic perturbation scenarios"""

    def __init__(self, model_name: str = "gemini-2.0-flash-lite"):
        super().__init__(model_name)
        self.operation_catalog = OperationCatalog()
        self.verifier = LLMOutputVerifier()

    def generate_scenario_specs(
        self, seed_trajectory: SeedTrajectory, window_states: List[Any], curriculum_config: CurriculumConfig
    ) -> List[ScenarioSpec]:
        """Generate diverse and strategic scenario specifications using LLM with full operation awareness"""
        task_context = self._analyze_task_context_with_llm(seed_trajectory, window_states, curriculum_config)
        scenarios = self._generate_diverse_scenarios_with_llm(task_context, curriculum_config.scenario_count)
        validated_scenarios = self._validate_scenarios(scenarios, task_context, seed_trajectory.task_id)
        diverse_scenarios = self._ensure_curriculum_diversity(validated_scenarios, task_context)
        prioritized_scenarios = self._prioritize_scenarios(diverse_scenarios, task_context)

        return prioritized_scenarios[: curriculum_config.scenario_count]

    def _generate_diverse_scenarios_with_llm(
        self, task_context: Dict[str, Any], scenario_count: int
    ) -> List[Dict[str, Any]]:
        """Generate scenarios with explicit diversity constraints"""
        max_retries = 3

        for attempt in range(max_retries):
            try:
                prompt = self._create_diverse_curriculum_prompt(task_context, scenario_count)
                response = self.call_llm(prompt)
                scenarios_data = self.extract_json(response)

                if scenarios_data:
                    return scenarios_data
                else:
                    if attempt < max_retries - 1:
                        self.logger.warning(
                            f"LLM diverse scenario generation attempt {attempt + 1} failed: No scenarios generated, retrying..."
                        )
                        continue
                    else:
                        self.logger.error(
                            "LLM diverse scenario generation failed after 3 attempts: No scenarios generated"
                        )
                        return []

            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(
                        f"LLM diverse scenario generation attempt {attempt + 1} failed: {e}, retrying..."
                    )
                    continue
                else:
                    self.logger.error(f"LLM diverse scenario generation failed after 3 attempts: {e}")
                    return []

        return []

    def _create_diverse_curriculum_prompt(self, task_context: Dict[str, Any], scenario_count: int) -> str:
        """Create curriculum prompt with explicit diversity requirements"""
        # Debug: Log the structure of task_context for troubleshooting
        self.logger.debug(f"Task context keys: {list(task_context.keys())}")
        if "perturbation_opportunities" in task_context:
            self.logger.debug(
                f"Perturbation opportunities count: {len(task_context['perturbation_opportunities'])}"
            )
            if task_context["perturbation_opportunities"]:
                self.logger.debug(
                    f"First opportunity keys: {list(task_context['perturbation_opportunities'][0].keys())}"
                )

        task_characteristics = task_context.get("task_characteristics", {})
        perturbation_opportunities = task_context.get("perturbation_opportunities", [])

        opportunities_text = ""
        if perturbation_opportunities:
            opportunities_text = "\nIDENTIFIED PERTURBATION OPPORTUNITIES:\n"
            for i, opp in enumerate(perturbation_opportunities, 1):
                # Use safe field access with better defaults
                element_type = opp.get("element_type", "ui_element")
                perturbation_type = opp.get("perturbation_type", "theme")
                timing = opp.get("timing", "runtime")
                intensity = opp.get("intensity", "medium")
                educational_value = opp.get("educational_value", "medium")
                risk_level = opp.get("risk_level", "low")

                opportunities_text += f"{i}. {element_type} - {perturbation_type} "
                opportunities_text += f"(timing: {timing}, intensity: {intensity}, "
                opportunities_text += f"educational_value: {educational_value}, risk: {risk_level})\n"

        characteristics_text = ""
        if task_characteristics:
            characteristics_text = "\nTASK CHARACTERISTICS:\n"
            for key, value in task_characteristics.items():
                characteristics_text += f"- {key}: {value}\n"

        prompt = f"""
{PROMPT_CONSTANTS["curriculum_role"].format(scenario_count=scenario_count)}

        TASK CONTEXT:
- Task: {task_context["instruction"]}
- Complexity: {task_context["complexity"]}
- Domain: {task_context["domain"]}
- Learning Objectives: {", ".join(task_context["learning_objectives"])}
- Target Applications: {", ".join(task_context["app_types"])}
{characteristics_text}{opportunities_text}

AVAILABLE OPERATIONS:
{task_context["available_operations"]}

MANDATORY DIVERSITY REQUIREMENTS:
{chr(10).join([f"{i + 1}. {req.format(categories=', '.join([pc.value for pc in PerturbationCategory]), types=', '.join([pt.value for pt in PerturbationType]), intensities=', '.join([pi.value for pi in PerturbationIntensity]))}" for i, req in enumerate(PROMPT_CONSTANTS["diversity_requirements"])])}

DIVERSITY CHECKLIST (ensure each scenario differs in at least 2 dimensions):
- Perturbation category: {"|".join([pc.value for pc in PerturbationCategory])}
- Perturbation type: {"|".join([pt.value for pt in PerturbationType])}
- Intensity: {"|".join([pi.value for pi in PerturbationIntensity])}
- Timing: {PROMPT_CONSTANTS["timing_options"]}
- Target scope: {PROMPT_CONSTANTS["target_scope_options"]}

Return JSON array with EXACTLY {scenario_count} scenario objects:
{{
            "target_app": "specific_app_name",
            "perturbation_trigger": "specific_condition_when_to_apply",
    "available_perturbation_actions": "simple_command_description_using_available_operations",
            "learning_objectives": "specific_learning_goal_for_visual_invariance",
            "target_components": ["specific_ui_elements_to_target"],
    "perturbation_types": ["{"|".join([pt.value for pt in PerturbationType])}"],
    "perturbation_category": "{"|".join([pc.value for pc in PerturbationCategory])}",
    "perturbation_intensity": "{"|".join([pi.value for pi in PerturbationIntensity])}",
            "maintains_functionality": true,
    "maintains_accessibility": true,
    "realistic_scenario": "brief_explanation_of_realistic_context",
    "initial_state_perturbation": true/false,
    "runtime_perturbation": true/false,
    "risk_mitigation": "brief_explanation_of_safety_measures",
    "educational_rationale": "brief_explanation_of_learning_value"
}}

CRITICAL REQUIREMENTS:
1. Each scenario must be UNIQUE and cover different aspects of visual invariance learning
2. available_perturbation_actions should be a simple string description, NOT complex JSON
3. All text fields should be brief but descriptive (1-2 sentences max)
4. Focus on perturbations that maintain target element accessibility
"""
        return prompt

    def _ensure_curriculum_diversity(
        self, scenarios: List[ScenarioSpec], task_context: Dict[str, Any]
    ) -> List[ScenarioSpec]:
        """Ensure curriculum diversity by checking and adjusting scenario distribution"""
        if len(scenarios) <= 1:
            return scenarios

        # Analyze current diversity
        categories_used = set()
        types_used = set()
        intensities_used = set()

        for scenario in scenarios:
            categories_used.add(scenario.perturbation_category.value)
            types_used.update([pt.value for pt in scenario.perturbation_types])
            intensities_used.add(scenario.perturbation_intensity.value)

        # Log diversity analysis
        self.logger.info("Curriculum diversity analysis:")
        self.logger.info(f"  Categories covered: {len(categories_used)}/{len(PerturbationCategory)}")
        self.logger.info(f"  Types covered: {len(types_used)}/{len(PerturbationType)}")
        self.logger.info(f"  Intensities covered: {len(intensities_used)}/{len(PerturbationIntensity)}")

        return scenarios

    def _analyze_task_context_with_llm(
        self, seed_trajectory: SeedTrajectory, window_states: List[Any], curriculum_config: CurriculumConfig
    ) -> Dict[str, Any]:
        """Analyze task and create comprehensive context for LLM using LLM-driven analysis only"""
        task_instruction = seed_trajectory.task_instruction
        llm_analysis = self._get_llm_task_analysis_with_retries(task_instruction, window_states)

        task_analysis = {
            "instruction": task_instruction,
            "complexity": llm_analysis.get("complexity", "moderate"),
            "domain": llm_analysis.get("domain", "general"),
            "learning_objectives": llm_analysis.get(
                "learning_objectives", ["Learn visual invariance across different UI states"]
            ),
            "app_types": [window_state.app_name for window_state in normalize_window_states(window_states)],
            "available_operations": self.operation_catalog.format_operations_for_llm(window_states),
            "scenario_count": curriculum_config.scenario_count,
            "task_characteristics": llm_analysis.get("task_characteristics", {}),
            "perturbation_opportunities": llm_analysis.get("perturbation_opportunities", []),
        }

        return task_analysis

    def _get_llm_task_analysis_with_retries(
        self, task_instruction: str, window_states: List[Any]
    ) -> Dict[str, Any]:
        """Use LLM to intelligently analyze task characteristics with retry mechanism"""

        try:
            app_states_summary = self._format_window_states_for_analysis(window_states)
            app_manipulation_analysis = self._analyze_app_manipulation_capabilities(window_states)

            prompt = f"""
{PROMPT_CONSTANTS["task_analysis_role"]}

TASK INSTRUCTION:
"{task_instruction}"

CURRENT APP STATES:
{app_states_summary}

APP MANIPULATION CAPABILITIES ANALYSIS:
{app_manipulation_analysis}

AVAILABLE OPERATIONS:
{self.operation_catalog.format_operations_for_llm(window_states)}

ANALYSIS REQUIREMENTS:
{chr(10).join(PROMPT_CONSTANTS["analysis_requirements"])}

EXAMPLES OF GOOD ANALYSIS:
{chr(10).join([f"- {example}" for example in PROMPT_CONSTANTS["good_analysis_examples"]])}

Return JSON:
{{
    "complexity": "{PROMPT_CONSTANTS["complexity_options"]}",
    "domain": "{PROMPT_CONSTANTS["domain_options"]}",
    "learning_objectives": [
        "specific_learning_goal_1",
        "specific_learning_goal_2",
        "specific_learning_goal_3"
    ],
    "task_characteristics": {{
        "estimated_steps": "number_or_range",
        "primary_apps": ["app1", "app2"],
        "critical_elements": ["element_type_1", "element_type_2"],
        "workflow_type": "{PROMPT_CONSTANTS["workflow_type_options"]}"
    }},
    "perturbation_opportunities": [
        {{
            "element_type": "critical_element_type",
            "perturbation_category": "{"|".join([pc.value for pc in PerturbationCategory])}",
            "perturbation_type": "{"|".join([pt.value for pt in PerturbationType])}",
            "target_scope": "{PROMPT_CONSTANTS["target_scope_options"]}",
            "timing": "{PROMPT_CONSTANTS["timing_options"]}",
            "intensity": "{"|".join([pi.value for pi in PerturbationIntensity])}",
            "educational_value": "{PROMPT_CONSTANTS["educational_value_options"]}",
            "risk_level": "{PROMPT_CONSTANTS["risk_level_options"]}",
            "maintains_accessibility": true/false
        }}
    ],
    "reasoning": "detailed_explanation_of_analysis"
        }}
        """

            response = self.call_llm(prompt)
            result = self.extract_json(response)

            if isinstance(result, list) and len(result) > 0:
                result = result[0]
                validated_result = self._validate_llm_task_analysis(result)
                if validated_result:
                    return validated_result
                else:
                    self.logger.exception("LLM task analysis result validation failed")

        except Exception as e:
            self.logger.exception(f"LLM task analysis failed after 3 attempts: {e}")
            return {}

    def _format_window_states_for_analysis(self, window_states: List[Any]) -> str:
        """Format window states for LLM task analysis with granular element details"""
        if not window_states:
            return "No window states available"

        normalized_states = normalize_window_states(window_states)
        formatted = []

        for window_state in normalized_states:
            app_name = window_state.app_name
            elements = window_state.get_all_elements()

            if not elements:
                formatted.append(f"App: {app_name} (no elements)")
                continue

            normalized_elements = normalize_ui_elements(elements)

            # Group elements by type and show names
            element_types = {}
            for element in normalized_elements:
                elem_type = element.element_type
                elem_name = element.name or "unnamed"

                if elem_type not in element_types:
                    element_types[elem_type] = []
                element_types[elem_type].append(elem_name)

            # Format detailed element summary
            app_summary = f"App: {app_name} ({len(elements)} elements):\n"
            for elem_type, elem_names in element_types.items():
                # Show first few element names for each type
                if len(elem_names) <= 3:
                    names_str = ", ".join(elem_names)
                else:
                    names_str = ", ".join(elem_names[:3]) + f" (and {len(elem_names) - 3} more)"

                app_summary += f"  - {elem_type}: {names_str}\n"

            formatted.append(app_summary.strip())

        return "\n".join(formatted)

    def _analyze_app_manipulation_capabilities(self, app_states: List[Any]) -> str:
        """Analyze app manipulation capabilities to guide perturbation strategy"""
        if not app_states:
            return "No app states available for analysis"

        normalized_states = normalize_window_states(app_states)
        analysis_parts = []

        for window_state in normalized_states:
            app_name = window_state.app_name
            app_ops = self.operation_catalog.get_operations_for_app(app_name)

            if not app_ops:
                analysis_parts.append(
                    f"{app_name.upper()}: Limited GUI manipulation - focus on system-level and content randomization"
                )
                continue

            gui_ops_count = sum(len(ops) for ops in app_ops.values())
            has_visual_ops = any(
                "ui" in category.lower() or "theme" in category.lower() for category in app_ops.keys()
            )

            if gui_ops_count < 10 or not has_visual_ops:
                analysis_parts.append(
                    f"{app_name.upper()}: Limited GUI manipulation capabilities - prioritize system-level perturbations"
                )
            else:
                analysis_parts.append(
                    f"{app_name.upper()}: Rich GUI manipulation capabilities - can use app-specific perturbations"
                )

        analysis_parts.append("\nPERTURBATION STRATEGY GUIDANCE:")
        analysis_parts.append(
            "- Apps with limited GUI manipulation: Focus on system themes, wallpapers, desktop layout, background processes, new apps or windows as distraction"
        )
        analysis_parts.append(
            "- Apps with rich GUI visual manipulation: Can use both app-specific and system-level perturbations"
        )
        analysis_parts.append(
            "- All apps: Consider content/data randomization (file contents, media properties, configurations)"
        )
        analysis_parts.append(
            "- Cross-app interference: Background notifications, competing windows, system resource changes"
        )

        return "\n".join(analysis_parts)

    def _validate_llm_task_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize LLM task analysis results"""
        validated = {}

        # Validate complexity
        complexity = analysis.get("complexity", "").lower()
        if complexity in ["simple", "moderate", "complex"]:
            validated["complexity"] = complexity
        else:
            validated["complexity"] = "moderate"  # Safe default

        # Validate domain
        domain = analysis.get("domain", "").lower()
        valid_domains = ["office", "web", "multimedia", "development", "system", "general"]
        if domain in valid_domains:
            validated["domain"] = domain
        else:
            validated["domain"] = "general"  # Safe default

        # Validate learning objectives
        objectives = analysis.get("learning_objectives", [])
        if isinstance(objectives, list) and len(objectives) > 0:
            validated["learning_objectives"] = [str(obj).strip() for obj in objectives if str(obj).strip()]
        else:
            validated["learning_objectives"] = ["Learn visual invariance across different UI states"]

        # Validate task characteristics
        characteristics = analysis.get("task_characteristics", {})
        if isinstance(characteristics, dict):
            validated["task_characteristics"] = characteristics
        else:
            validated["task_characteristics"] = {}

        # Validate perturbation opportunities
        opportunities = analysis.get("perturbation_opportunities", [])
        if isinstance(opportunities, list):
            validated["perturbation_opportunities"] = opportunities
        else:
            validated["perturbation_opportunities"] = []

        return validated

    def _validate_scenarios(
        self, scenarios: List[Dict[str, Any]], task_context: Dict[str, Any], task_id: str
    ) -> List[ScenarioSpec]:
        """Validate and convert scenarios to ScenarioSpec objects"""
        validated_scenarios = []

        for i, scenario_data in enumerate(scenarios):
            try:
                sanitized_data = self.verifier.sanitize_scenario_data(scenario_data)
                is_valid, errors = self.verifier.verify_scenario_spec(sanitized_data)

                if not is_valid:
                    self.logger.warning(f"Scenario {i} validation failed: {errors}")
                    continue

                scenario_spec = self._convert_to_scenario_spec(sanitized_data, task_id, i)
                if scenario_spec:
                    validated_scenarios.append(scenario_spec)

            except Exception as e:
                self.logger.error(f"Error processing scenario {i}: {e}")
                continue

        return validated_scenarios

    def _convert_to_scenario_spec(
        self, scenario_data: Dict[str, Any], task_id: str, scenario_index: int
    ) -> Optional[ScenarioSpec]:
        """Convert scenario data to ScenarioSpec object"""
        try:
            # Parse perturbation types
            perturbation_types = []
            for pt_str in scenario_data.get("perturbation_types", []):
                # Types are already mapped to valid enum values by verifier
                try:
                    perturbation_types.append(PerturbationType(pt_str))
                except ValueError:
                    perturbation_types.append(PerturbationType.THEME)

            if not perturbation_types:
                perturbation_types.append(PerturbationType.THEME)

            # Generate meaningful scenario ID: task_id + scenario_number + target_app
            target_app = scenario_data.get("target_app", "system")
            scenario_id = f"{task_id}_scenario_{scenario_index + 1}_{target_app}"

            return ScenarioSpec(
                scenario_id=scenario_id,
                target_app=scenario_data.get("target_app", "system"),
                perturbation_trigger=scenario_data.get("perturbation_trigger", "During task execution"),
                available_perturbation_actions=scenario_data.get("available_perturbation_actions", ""),
                learning_objectives=scenario_data.get("learning_objectives", "Learn visual invariance"),
                target_components=scenario_data.get("target_components", []),
                perturbation_types=perturbation_types,
                perturbation_category=PerturbationCategory.from_string(
                    scenario_data.get("perturbation_category", "system_level")
                ),
                perturbation_intensity=PerturbationIntensity.from_string(
                    scenario_data.get("perturbation_intensity", "medium")
                ),
                maintains_functionality=scenario_data.get("maintains_functionality", True),
                maintains_accessibility=scenario_data.get("maintains_accessibility", True),
                realistic_scenario=scenario_data.get("realistic_scenario", ""),
                initial_state_perturbation=scenario_data.get("initial_state_perturbation", False),
                runtime_perturbation=scenario_data.get("runtime_perturbation", True),
                risk_mitigation=scenario_data.get("risk_mitigation", ""),
                educational_rationale=scenario_data.get("educational_rationale", ""),
            )

        except Exception as e:
            self.logger.error(f"Error converting scenario: {e}")
            return None

    def _prioritize_scenarios(
        self, scenarios: List[ScenarioSpec], task_context: Dict[str, Any]
    ) -> List[ScenarioSpec]:
        """Prioritize scenarios based on learning objectives and task relevance"""

        def scenario_priority(scenario):
            priority = 0

            # Higher priority for task-relevant apps
            if scenario.target_app.lower() in [app.lower() for app in task_context["app_types"]]:
                priority += 3

            # Higher priority for diverse perturbation types
            if len(scenario.perturbation_types) > 1:
                priority += 2

            # Higher priority for specific learning objectives
            if "visual invariance" in scenario.learning_objectives.lower():
                priority += 2

            return priority

        return sorted(scenarios, key=scenario_priority, reverse=True)


class CleanPerturbationGenerator(CleanLLM):
    """LLM-driven perturbation generator with comprehensive operation awareness and procedural memory"""

    def __init__(self, model_name: str = "gemini-2.0-flash-lite"):
        super().__init__(model_name)
        self.operation_catalog = OperationCatalog()
        self.procedural_memory = ProceduralMemory()
        self.verifier = LLMOutputVerifier()

    def decide_perturbation(
        self, execution_context: ExecutionContext, scenario_spec: ScenarioSpec
    ) -> Dict[str, Any]:
        """Decide whether to apply perturbation with procedural memory context"""
        llm_decision = self._get_llm_decision_with_context(execution_context, scenario_spec)

        if llm_decision.get("should_apply", False):
            enhanced_decision = self._enhance_with_procedural_memory(
                llm_decision, scenario_spec, execution_context
            )
            return enhanced_decision

        return llm_decision

    def _get_llm_decision_with_context(
        self, execution_context: ExecutionContext, scenario_spec: ScenarioSpec
    ) -> Dict[str, Any]:
        """Get LLM decision with procedural memory context and retries"""

        try:
            memory_context = self.procedural_memory.get_coherence_context(scenario_spec.target_app)

            # Get available operations for target app
            mock_app_state = type("MockAppState", (), {"app_name": scenario_spec.target_app})()
            app_operations = self.operation_catalog.format_operations_for_llm([mock_app_state])

            prompt = f"""
{PROMPT_CONSTANTS["perturbation_role"]}

        CURRENT EXECUTION CONTEXT:
        Step: {execution_context.step_idx}
        Action: {execution_context.current_action}
        Task: {execution_context.task_instruction}
        App States: {self._format_app_states_for_decision(execution_context.window_states)}

        SCENARIO SPECIFICATION:
        Target App: {scenario_spec.target_app}
        Trigger: {scenario_spec.perturbation_trigger}
        Available Actions: {scenario_spec.available_perturbation_actions}
        Learning Objectives: {scenario_spec.learning_objectives}
        Perturbation Types: {[pt.value for pt in scenario_spec.perturbation_types]}

AVAILABLE OPERATIONS FOR {scenario_spec.target_app.upper()}:
{app_operations}

        PROCEDURAL MEMORY CONTEXT:
        {memory_context}

        PERTURBATION DECISION CRITERIA:
{chr(10).join([f"{i + 1}. {criteria}" for i, criteria in enumerate(PROMPT_CONSTANTS["perturbation_criteria"])])}

DECISION EXAMPLES:
{chr(10).join([f"- {example}" for example in PROMPT_CONSTANTS["decision_examples"]])}

COMMAND GENERATION EXAMPLES:
{chr(10).join([f"- {example}" for example in PROMPT_CONSTANTS["command_examples"]])}

        Return JSON:
        {{
            "should_apply": true/false,
            "reasoning": "detailed_explanation_of_decision",
    "perturbation_type": "{"|".join([pt.value for pt in PerturbationType])}",
    "api_call": "execute_bash_command|execute_python_command|execute_css_injection|execute_dom_modification|execute_theme_randomization|execute_system_perturbation",
    "generated_command": "concrete_perturbation_command_using_available_operations",
            "parameters": {{
                "target_app": "{scenario_spec.target_app}",
                "intensity": "low|medium|high",
                "coherent_with_history": true/false,
        "maintains_functionality": true/false,
        "preserves_target_accessibility": true/false
            }},
            "confidence": 0.0-1.0,
    "alternative_commands": ["list_of_alternative_perturbation_commands"]
        }}
        """

            response = self.call_llm(prompt)
            result = self.extract_json(response)

            if isinstance(result, list) and len(result) > 0:
                result = result[0]

            if not isinstance(result, dict):
                self.logger.error("Invalid response format from LLM")
                return {
                    "should_apply": False,
                    "reasoning": "Failed to parse LLM response",
                    "perturbation_type": "theme",
                    "generated_command": "",
                    "parameters": {"target_app": scenario_spec.target_app},
                    "confidence": 0.0,
                    "alternative_commands": [],
                }

            is_valid, errors = self.verifier.verify_perturbation_decision(result)
            if not is_valid:
                self.logger.error(f"Validation failed: {', '.join(errors)}")
                return {
                    "should_apply": False,
                    "reasoning": f"Validation failed: {', '.join(errors)}",
                    "perturbation_type": "theme",
                    "generated_command": "",
                    "parameters": {"target_app": scenario_spec.target_app},
                    "confidence": 0.0,
                    "alternative_commands": [],
                }

            return result

        except Exception as e:
            self.logger.error(f"Error in _get_llm_decision_with_context: {e}")
            return {
                "should_apply": False,
                "reasoning": f"LLM call failed: {e}",
                "perturbation_type": "theme",
                "generated_command": "",
                "parameters": {"target_app": scenario_spec.target_app},
                "confidence": 0.0,
                "alternative_commands": [],
            }

    def _format_app_states_for_decision(self, app_states: List[Any]) -> str:
        """Format app states for perturbation decision"""
        if not app_states:
            return "No app states available"

        normalized_states = normalize_window_states(app_states)
        formatted = []

        for window_state in normalized_states:
            app_name = window_state.app_name
            elements = window_state.get_all_elements()

            if not elements:
                formatted.append(f"App: {app_name} (no elements detected)")
                continue

            normalized_elements = normalize_ui_elements(elements)

            interactive_elements = [
                elem
                for elem in normalized_elements
                if elem.element_type.lower()
                in ["button", "link", "input", "menu", "checkbox", "radio", "slider", "combo-box"]
            ]

            # Group elements by type for better context
            element_types = {}
            for elem in normalized_elements:
                elem_type = elem.element_type
                if elem_type not in element_types:
                    element_types[elem_type] = 0
                element_types[elem_type] += 1

            # Format element summary
            type_summary = ", ".join([f"{count} {elem_type}" for elem_type, count in element_types.items()])

            formatted.append(f"App: {app_name} ({len(elements)} total elements: {type_summary})")

            # Show key interactive elements
            if interactive_elements:
                key_elements = interactive_elements[:3]  # Show first 3
                element_names = [elem.name or elem.element_type for elem in key_elements]
                formatted.append(f"  Key interactive: {', '.join(element_names)}")

        return "\n".join(formatted)

    def _get_diverse_curriculum_examples(self, task_context: Dict[str, Any], scenario_count: int) -> str:
        """Generate diverse curriculum examples to avoid bias and ensure variety"""
        _domain = task_context.get("domain", "general")
        app_types = task_context.get("app_types", [])

        # Define curriculum diversity dimensions
        perturbation_categories = [pc.value for pc in PerturbationCategory]
        perturbation_types = [pt.value for pt in PerturbationType]
        _intensities = [pi.value for pi in PerturbationIntensity]
        _target_scopes = ["system", "app", "file", "content"]
        _timings = ["initial", "runtime", "between_steps"]

        # Generate diverse examples based on curriculum requirements
        examples = []

        # Ensure we cover different perturbation categories
        for i, category in enumerate(perturbation_categories[:scenario_count]):
            if category == "system_level":
                examples.append(
                    f"System-level: change_wallpaper('/path/to/theme_{i}.jpg') - {category} perturbation"
                )
            elif category == "content_randomization":
                examples.append(
                    f"Content: modify_file_content('/path/to/file_{i}', 'randomized_content') - {category} perturbation"
                )
            elif category == "app_specific":
                if "vlc" in [app.lower() for app in app_types]:
                    examples.append(
                        f"VLC-specific: VLCTools.set_settings('setting_{i}', 'value_{i}') - {category} perturbation"
                    )
                elif "chrome" in [app.lower() for app in app_types]:
                    examples.append(
                        f"Chrome-specific: execute_css_injection('rule_{i} {{ property: value_{i}; }}') - {category} perturbation"
                    )
                else:
                    examples.append(
                        f"App-specific: execute_app_command('command_{i}') - {category} perturbation"
                    )
            elif category == "cross_app_interference":
                examples.append(
                    f"Cross-app: execute_bash_command('launch_app_{i} && sleep 1') - {category} perturbation"
                )

        # Add variety in perturbation types
        type_examples = []
        for i, ptype in enumerate(perturbation_types[:3]):  # Limit to avoid overwhelming
            if ptype == "theme":
                type_examples.append(f"Theme variation: Change visual theme to variant_{i}")
            elif ptype == "layout":
                type_examples.append("Layout variation: Modify element positioning and spacing")
            elif ptype == "content_variation":
                type_examples.append("Content variation: Randomize data properties and values")
            elif ptype == "system_level":
                type_examples.append("System-level: Modify desktop environment settings")

        # Combine and limit examples
        all_examples = examples + type_examples
        return "\n".join([f"- {example}" for example in all_examples[:scenario_count]])

    def _enhance_with_procedural_memory(
        self, llm_decision: Dict[str, Any], scenario_spec: ScenarioSpec, execution_context: ExecutionContext
    ) -> Dict[str, Any]:
        """Enhance LLM decision with procedural memory"""
        try:
            target_app = scenario_spec.target_app.lower()
            perturbation_type = llm_decision.get("perturbation_type", "theme")
            generated_command = llm_decision.get("generated_command", "")

            if generated_command:
                # Update procedural memory
                self.procedural_memory.add_perturbation(
                    execution_context.step_idx,
                    target_app,
                    perturbation_type,
                    generated_command,
                    execution_context.window_states[0] if execution_context.window_states else {},
                )

                llm_decision["procedural_memory_enhanced"] = True
                llm_decision["reasoning"] += " (Enhanced with procedural memory)"

            return llm_decision

        except Exception as e:
            self.logger.error(f"Error enhancing with procedural memory: {e}")
            return llm_decision


class CleanElementIdentificationLLM(CleanLLM):
    """Clean element identification using LLM"""

    def identify_target_element(
        self, action_str: str, app_states: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Use LLM to identify the target element from action string and app states"""
        candidates = self.identify_target_element_candidates(action_str, app_states)
        return candidates[0] if candidates else None

    def identify_target_element_candidates(
        self, action_str: str, window_states: List[WindowState]
    ) -> List[Dict[str, Any]]:
        """Use LLM to identify ALL potential target elements from action string and window states"""

        window_states_summary = self._format_window_states_for_llm(window_states)

        prompt = f"""
        Find ALL possible UI elements that this action could be trying to interact with.
        Return them ranked by likelihood, with the most likely first.

        Action: "{action_str}"

        Available Elements:
        {window_states_summary}

        IMPORTANT DISAMBIGUATION RULES:
        - If multiple elements have the same name, consider ALL of them as potential candidates
        - Prioritize elements that are likely visible and interactive based on coordinates and type
        - Elements with coordinates (0,0) or very small sizes are likely hidden/invisible
        - Elements with coordinates outside screen bounds (0-1920, 0-1080) are likely invalid
        - Elements marked as "[collapsed menu - likely invisible]" or "[hidden dropdown - likely invisible]" should be deprioritized
        - Elements marked as "[inactive tab - likely invisible]" should be deprioritized
        - Elements marked as "[blocked by higher window]" should be deprioritized
        - Rank by: 1) Element type appropriateness, 2) Coordinate reasonableness, 3) Hierarchy visibility, 4) Context relevance

        Return JSON array with element identifiers, ranked by likelihood from 0.00 to 1.00 and only return elements with confidence values greater than 0.70:
        [
            {{
                "name": "element_name",
                "element_type": "element_type",
                "app_name": "app_name",
                "confidence": 0.95,
                "reasoning": "detailed_reasoning_for_element_selection_including_coordinate_analysis"
            }},
            {{
                "name": "alternative_element_name",
                "element_type": "element_type",
                "app_name": "app_name",
                "confidence": 0.75,
                "reasoning": "alternative_reasoning_with_coordinate_considerations"
            }}
        ]
        """

        response = self.call_llm(prompt)
        result = self.extract_json(response)

        if not isinstance(result, list):
            return []

        # Validate and filter results
        valid_candidates = []
        for candidate in result:
            if not isinstance(candidate, dict):
                continue

            # Validate required fields
            required_fields = ["name", "element_type", "app_name"]
            if not all(field in candidate for field in required_fields):
                continue

            if candidate.get("name") is None:
                continue

            valid_candidates.append(candidate)

        return valid_candidates

    def _format_window_states_for_llm(self, window_states: List[WindowState]) -> str:
        """Format window states with complete hierarchical element tree for LLM consumption"""
        if not window_states:
            return "No window states available"

        normalized_states = normalize_window_states(window_states)
        formatted_states = []

        for window_state in normalized_states:
            app_name = window_state.app_name
            window_name = window_state.window_name

            # Show window hierarchy information
            if window_name != app_name:
                app_summary = f"App: {app_name} - Window: {window_name}\n"
            else:
                app_summary = f"App: {app_name}\n"

            # Add window properties if available
            if window_state.is_active:
                app_summary += "  [TOP WINDOW]\n"
            if window_state.z_order > 500:
                app_summary += "  [HIGH PRIORITY WINDOW]\n"

            # Traverse the hierarchical element tree with z-order blocking consideration
            if window_state.root_element:
                app_summary += self._format_element_tree_hierarchical(
                    window_state.root_element,
                    depth=0,
                    window_state=window_state,
                    all_window_states=normalized_states,
                )
            else:
                app_summary += "  (no elements detected)\n"

            formatted_states.append(app_summary)

        return "\n".join(formatted_states)

    def _format_app_states_for_llm(self, app_states: List[Any]) -> str:
        """Format app states with hierarchical element tree for LLM consumption"""
        if not app_states:
            return "No app states available"

        normalized_states = normalize_window_states(app_states)
        formatted_states = []

        for window_state in normalized_states:
            app_name = window_state.app_name
            window_name = window_state.window_name

            # Show window hierarchy information
            if window_name != app_name:
                app_summary = f"App: {app_name} - Window: {window_name}\n"
            else:
                app_summary = f"App: {app_name}\n"

            # Add window properties if available
            if window_state.is_active:
                app_summary += "  [TOP WINDOW]\n"
            if window_state.z_order > 500:
                app_summary += "  [HIGH PRIORITY WINDOW]\n"

            # Traverse the hierarchical element tree with z-order blocking consideration
            if window_state.root_element:
                app_summary += self._format_element_tree_hierarchical(
                    window_state.root_element,
                    depth=0,
                    window_state=window_state,
                    all_window_states=normalized_states,
                )
            else:
                app_summary += "  (no elements detected)\n"

            formatted_states.append(app_summary)

        return "\n".join(formatted_states)

    def _format_element_tree_hierarchical(
        self,
        element: UIElement,
        depth: int = 0,
        window_state: WindowState = None,
        all_window_states: List[WindowState] = None,
    ) -> str:
        """Format element tree with complete hierarchical structure and visibility context"""
        if not element:
            return ""

        result = ""
        indent = "  " * (depth + 1)  # +1 because we're inside the window

        # Show ALL elements in the tree (autoglm_integration.py already filtered hidden parents)
        # Add context hints based on element properties and z-order blocking
        context_hints = self._get_element_context_hints(element, window_state, all_window_states)

        # Format element name and type
        display_name = element.name if element.name else f"{element.element_type}"

        # Format position
        pos = element.position
        position_str = f"at ({pos.get('center_x', 0)}, {pos.get('center_y', 0)})" if pos else "no position"

        # Add element line
        result += f"{indent}- {display_name} ({element.element_type}){context_hints} {position_str}\n"

        # Traverse all children since autoglm_integration.py already filtered out hidden parent children
        for child in element.children:
            child_result = self._format_element_tree_hierarchical(
                child, depth + 1, window_state, all_window_states
            )
            result += child_result

        return result

    def _should_show_element(
        self,
        element: UIElement,
        window_state: WindowState = None,
        all_window_states: List[WindowState] = None,
    ) -> bool:
        """Determine if element should be shown based on visibility and z-order blocking"""
        # Always show structural elements (they provide context)
        if element.element_type in ["frame", "panel", "filler", "layered-pane"]:
            return True

        # Show visible elements (autoglm_integration.py already filtered out children of hidden parents)
        if element.visibility.value == "visible":
            # Check if element is blocked by higher z-order windows
            if self._is_element_blocked_by_z_order(element, window_state, all_window_states):
                return False
            return True

        # Show elements that might be interactive even if not fully visible
        if element.element_type in [
            "button",
            "link",
            "input",
            "menu",
            "checkbox",
            "radio",
            "slider",
            "combo-box",
        ]:
            # Still check z-order blocking for interactive elements
            if self._is_element_blocked_by_z_order(element, window_state, all_window_states):
                return False
            return True

    def _is_element_blocked_by_z_order(
        self, element: UIElement, window_state: WindowState, all_window_states: List[WindowState]
    ) -> bool:
        """Check if element is blocked by a higher z-order window"""
        if not element.position or not window_state or not all_window_states:
            return False

        element_x = element.position.get("x", 0)
        element_y = element.position.get("y", 0)
        element_width = element.position.get("width", 0)
        element_height = element.position.get("height", 0)

        if element_width <= 0 or element_height <= 0:
            return False

        # Check all windows with higher z-order than current window
        current_z_order = window_state.z_order
        for other_window in all_window_states:
            if other_window.z_order > current_z_order and other_window.is_mapped:
                # Check if other window's geometry overlaps with element
                other_geometry = other_window.geometry
                if not other_geometry:
                    continue

                other_x = other_geometry.get("x", 0)
                other_y = other_geometry.get("y", 0)
                other_width = other_geometry.get("width", 0)
                other_height = other_geometry.get("height", 0)

                # Check for overlap
                if (
                    other_x < element_x + element_width
                    and other_x + other_width > element_x
                    and other_y < element_y + element_height
                    and other_y + other_height > element_y
                ):
                    return True

        return False

    def _get_element_context_hints(
        self,
        element: UIElement,
        window_state: WindowState = None,
        all_window_states: List[WindowState] = None,
    ) -> str:
        """Get context hints for element based on its properties and z-order blocking"""
        hints = []

        # Element type hints
        if "menu-item" in element.element_type:
            hints.append("[menu]")
        elif "check-box" in element.element_type:
            hints.append("[checkbox]")
        elif "button" in element.element_type:
            hints.append("[button]")
        elif "tab" in element.element_type:
            hints.append("[tab]")
        elif "input" in element.element_type:
            hints.append("[input]")
        elif "link" in element.element_type:
            hints.append("[link]")

        # Visibility hints (simplified since autoglm_integration.py handles parent filtering)
        if element.visibility.value == "collapsed":
            hints.append("[collapsed - likely invisible]")
        elif element.visibility.value == "hidden_window":
            hints.append("[hidden window - likely invisible]")
        elif element.visibility.value == "hidden_tab":
            hints.append("[inactive tab - likely invisible]")
        elif element.visibility.value == "structural":
            hints.append("[structural container]")
        elif element.visibility.value == "not_showing":
            hints.append("[not showing - likely invisible]")

        # Z-order blocking hints
        if window_state and all_window_states:
            if self._is_element_blocked_by_z_order(element, window_state, all_window_states):
                hints.append("[blocked by higher window]")

        # State hints
        if element.is_focused:
            hints.append("[focused]")
        if not element.is_enabled:
            hints.append("[disabled]")
        if element.is_expanded:
            hints.append("[expanded]")

        # Depth hint for context
        if element.depth > 3:
            hints.append(f"[deep: {element.depth}]")

        return " " + " ".join(hints) if hints else ""


class CleanQualityLLM(CleanLLM):
    """Clean quality evaluation"""

    def evaluate_trajectory_quality(
        self, generated_trajectory: GeneratedTrajectory, scenario_spec: ScenarioSpec
    ) -> float:
        """Evaluate trajectory quality"""

        prompt = f"""
        Evaluate trajectory quality (0.0-1.0):

        Success: {generated_trajectory.success}
        Perturbations: {len(generated_trajectory.perturbation_log)}
        Learning Objectives: {scenario_spec.learning_objectives}

        Return JSON:
        {{
            "quality_score": 0.0-1.0,
            "reasoning": "explanation"
        }}
        """

        response = self.call_llm(prompt)
        result = self.extract_json(response)

        if isinstance(result, list) and len(result) > 0:
            result = result[0]

        return result.get("quality_score", 0.0) if isinstance(result, dict) else 0.0


class LLMOutputVerifier:
    """Comprehensive LLM output verification and parsing system"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def verify_scenario_spec(self, scenario_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Verify scenario specification completeness and validity"""
        errors = []

        required_fields = [
            "target_app",
            "perturbation_trigger",
            "available_perturbation_actions",
            "learning_objectives",
            "target_components",
            "perturbation_types",
            "perturbation_category",
        ]

        for field in required_fields:
            if field not in scenario_data:
                errors.append(f"Missing required field: {field}")
            elif not scenario_data[field]:
                errors.append(f"Empty required field: {field}")

        # Validate target_app
        if "target_app" in scenario_data:
            target_app = scenario_data["target_app"]
            if not isinstance(target_app, str) or target_app.lower() == "unknown":
                errors.append("Invalid target_app: must be specific application name")

        # Validate perturbation_types
        if "perturbation_types" in scenario_data:
            perturbation_types = scenario_data["perturbation_types"]
            if not isinstance(perturbation_types, list) or len(perturbation_types) == 0:
                errors.append("Invalid perturbation_types: must be non-empty list")
            else:
                # Use flexible mapping instead of strict validation
                mapped_types = []
                for pt_str in perturbation_types:
                    mapped_type = PerturbationType.from_string(pt_str, default=PerturbationType.THEME)
                    mapped_types.append(mapped_type.value)

                # Update the scenario data with mapped types
                scenario_data["perturbation_types"] = mapped_types

        # Validate target_components
        if "target_components" in scenario_data:
            target_components = scenario_data["target_components"]
            if not isinstance(target_components, list):
                errors.append("Invalid target_components: must be list")

        # Validate learning_objectives
        if "learning_objectives" in scenario_data:
            learning_objectives = scenario_data["learning_objectives"]
            if not isinstance(learning_objectives, str) or len(learning_objectives) < 10:
                errors.append("Invalid learning_objectives: must be descriptive string")

        # Validate perturbation_category
        if "perturbation_category" in scenario_data:
            perturbation_category = scenario_data["perturbation_category"]
            if not isinstance(perturbation_category, str):
                errors.append("Invalid perturbation_category: must be string")
            else:
                # Use flexible mapping instead of strict validation
                mapped_category = PerturbationCategory.from_string(
                    perturbation_category, default=PerturbationCategory.SYSTEM_LEVEL
                )
                scenario_data["perturbation_category"] = mapped_category.value

        return len(errors) == 0, errors

    def verify_perturbation_decision(self, decision_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Verify perturbation decision completeness and validity"""
        errors = []

        # Required fields
        required_fields = ["should_apply", "reasoning", "perturbation_type", "api_call", "parameters"]

        for field in required_fields:
            if field not in decision_data:
                errors.append(f"Missing required field: {field}")

        # Validate should_apply
        if "should_apply" in decision_data:
            should_apply = decision_data["should_apply"]
            if not isinstance(should_apply, bool):
                errors.append("Invalid should_apply: must be boolean")

        # Validate reasoning
        if "reasoning" in decision_data:
            reasoning = decision_data["reasoning"]
            if not isinstance(reasoning, str) or len(reasoning) < 5:
                errors.append("Invalid reasoning: must be descriptive string")

        # Validate perturbation_type
        if "perturbation_type" in decision_data:
            perturbation_type = decision_data["perturbation_type"]
            if not isinstance(perturbation_type, str):
                errors.append("Invalid perturbation_type: must be string")
            else:
                # Use flexible mapping instead of strict validation
                mapped_type = PerturbationType.from_string(perturbation_type, default=PerturbationType.THEME)
                decision_data["perturbation_type"] = mapped_type.value

        # Validate api_call
        if "api_call" in decision_data:
            api_call = decision_data["api_call"]
            if not isinstance(api_call, str):
                errors.append("Invalid api_call: must be string")
            else:
                # Use flexible mapping instead of strict validation
                mapped_api_call = ApiCallType.from_string(api_call, default=ApiCallType.EXECUTE_BASH_COMMAND)
                decision_data["api_call"] = mapped_api_call.value

        # Validate parameters
        if "parameters" in decision_data:
            parameters = decision_data["parameters"]
            if not isinstance(parameters, dict):
                errors.append("Invalid parameters: must be dictionary")

        return len(errors) == 0, errors

    def sanitize_scenario_data(self, scenario_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize and normalize scenario data"""
        sanitized = {}

        # Sanitize string fields
        string_fields = [
            "scenario_id",
            "target_app",
            "perturbation_trigger",
            "available_perturbation_actions",
            "learning_objectives",
            "perturbation_category",
        ]

        for field in string_fields:
            if field in scenario_data:
                value = scenario_data[field]
                if isinstance(value, str):
                    sanitized[field] = value.strip()
                else:
                    sanitized[field] = str(value).strip()

        # Sanitize list fields
        list_fields = ["target_components", "perturbation_types"]

        for field in list_fields:
            if field in scenario_data:
                value = scenario_data[field]
                if isinstance(value, list):
                    sanitized[field] = [str(item).strip() for item in value if item]
                else:
                    sanitized[field] = [str(value).strip()]

        # Sanitize boolean fields
        boolean_fields = ["maintains_functionality"]

        for field in boolean_fields:
            if field in scenario_data:
                value = scenario_data[field]
                if isinstance(value, bool):
                    sanitized[field] = value
                else:
                    sanitized[field] = bool(value)

        # Sanitize intensity
        if "perturbation_intensity" in scenario_data:
            intensity = scenario_data["perturbation_intensity"]
            valid_intensities = ["low", "medium", "high"]
            if intensity in valid_intensities:
                sanitized["perturbation_intensity"] = intensity
            else:
                sanitized["perturbation_intensity"] = "medium"

        return sanitized

    def enhance_scenario_with_defaults(self, scenario_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance scenario with default values for missing fields"""
        enhanced = scenario_data.copy()

        defaults = {
            "scenario_id": f"scenario_{hash(str(scenario_data))}",
            "target_app": "system",
            "perturbation_trigger": "During task execution",
            "available_perturbation_actions": 'echo "Default perturbation applied"',
            "learning_objectives": "Learn to adapt to visual changes",
            "target_components": ["general"],
            "perturbation_types": ["theme"],
            "perturbation_category": PerturbationCategory.SYSTEM_LEVEL.value,
            "maintains_functionality": True,
            "perturbation_intensity": PerturbationIntensity.MEDIUM.value,
            "realistic_scenario": "Generic perturbation scenario",
        }

        for key, default_value in defaults.items():
            if key not in enhanced or not enhanced[key]:
                enhanced[key] = default_value

        return enhanced
