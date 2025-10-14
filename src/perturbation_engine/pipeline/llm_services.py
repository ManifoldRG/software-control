"""
LLM Services: interfaces for LLM interactions with comprehensive operation awareness

Usage Examples:
    # Using Gemini (default)
    curriculum_gen = CurriculumGenerator()

    # Using OpenAI GPT models
    curriculum_gen = CurriculumGenerator(model_name="gpt-5-nano")        # Budget option
    curriculum_gen = CurriculumGenerator(model_name="gpt-4o-mini")       # Budget option
    curriculum_gen = CurriculumGenerator(model_name="gpt-5-mini")         # Mid-tier
    curriculum_gen = CurriculumGenerator(model_name="gpt-5")              # Premium
    curriculum_gen = CurriculumGenerator(model_name="gpt-4o")             # Premium

    # Using Anthropic Claude models
    curriculum_gen = CurriculumGenerator(model_name="claude-haiku-3.5")   # Mid-tier
    curriculum_gen = CurriculumGenerator(model_name="claude-sonnet-4")    # Premium

    # Using OpenRouter models
    curriculum_gen = CurriculumGenerator(model_name="openrouter-claude-3.5-sonnet")

Environment Variables Required:
    - GEMINI_API_KEY for Gemini models
    - OPENAI_API_KEY for OpenAI models
    - ANTHROPIC_API_KEY for Anthropic models
    - OPENROUTER_API_KEY for OpenRouter models
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
from perturbation_engine.pipeline.operation_examples import OperationExamples

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
    """Enhanced procedural memory for trajectory coherence and visual consistency"""

    def __init__(self):
        self.perturbation_history = []
        self.visual_state_tracker = {}  # Track visual states per app
        self.trajectory_patterns = {}  # Track trajectory-level patterns
        self.logger = logging.getLogger(__name__)

    def add_perturbation(
        self, step_idx: int, target_app: str, perturbation_type: str, command: str, app_state: Dict[str, Any]
    ):
        """Add perturbation to memory with enhanced tracking"""
        perturbation_data = {
            "step_idx": step_idx,
            "target_app": target_app,
            "perturbation_type": perturbation_type,
            "command": command,
            "app_state": app_state,
            "success": True,  # Default to success
            "visual_impact": self._extract_visual_impact(command, perturbation_type),
            "timestamp": step_idx,  # Use step as timestamp
        }

        self.perturbation_history.append(perturbation_data)

        # Update visual state tracking
        self._update_visual_state(target_app, perturbation_data)

        # Update trajectory patterns
        self._update_trajectory_patterns(perturbation_data)

    def _extract_visual_impact(self, command: str, perturbation_type: str) -> Dict[str, Any]:
        """Extract visual impact information from command"""
        visual_impact = {
            "theme_change": False,
            "color_change": False,
            "layout_change": False,
            "typography_change": False,
            "intensity": "low",
        }

        command_lower = command.lower()

        # Detect theme changes
        if "theme" in command_lower or "gtk-theme" in command_lower:
            visual_impact["theme_change"] = True
            visual_impact["intensity"] = "high"

        # Detect color changes
        if any(
            color_indicator in command_lower
            for color_indicator in ["color", "background", "border", "rgba", "#"]
        ):
            visual_impact["color_change"] = True
            visual_impact["intensity"] = "medium"

        # Detect layout changes
        if any(
            layout_indicator in command_lower
            for layout_indicator in ["margin", "padding", "size", "position"]
        ):
            visual_impact["layout_change"] = True
            visual_impact["intensity"] = "medium"

        # Detect typography changes
        if any(font_indicator in command_lower for font_indicator in ["font", "text", "typography"]):
            visual_impact["typography_change"] = True
            visual_impact["intensity"] = "low"

        return visual_impact

    def _update_visual_state(self, target_app: str, perturbation_data: Dict[str, Any]):
        """Update visual state tracking for an app"""
        if target_app not in self.visual_state_tracker:
            self.visual_state_tracker[target_app] = {
                "current_theme": "default",
                "color_scheme": "default",
                "visual_changes": [],
                "last_change_step": 0,
            }

        visual_state = self.visual_state_tracker[target_app]
        visual_impact = perturbation_data["visual_impact"]

        # Update theme tracking
        if visual_impact["theme_change"]:
            if "dark" in perturbation_data["command"].lower():
                visual_state["current_theme"] = "dark"
            elif "light" in perturbation_data["command"].lower():
                visual_state["current_theme"] = "light"

        # Track visual changes
        visual_state["visual_changes"].append(
            {
                "step": perturbation_data["step_idx"],
                "type": perturbation_data["perturbation_type"],
                "impact": visual_impact,
                "command_preview": perturbation_data["command"][:30],
            }
        )

        # Keep only last 5 visual changes
        visual_state["visual_changes"] = visual_state["visual_changes"][-5:]
        visual_state["last_change_step"] = perturbation_data["step_idx"]

    def _update_trajectory_patterns(self, perturbation_data: Dict[str, Any]):
        """Update trajectory-level patterns"""
        step_idx = perturbation_data["step_idx"]

        # Track perturbation frequency patterns
        if "perturbation_frequency" not in self.trajectory_patterns:
            self.trajectory_patterns["perturbation_frequency"] = []

        self.trajectory_patterns["perturbation_frequency"].append(
            {
                "step": step_idx,
                "app": perturbation_data["target_app"],
                "type": perturbation_data["perturbation_type"],
            }
        )

        # Track visual progression patterns
        if "visual_progression" not in self.trajectory_patterns:
            self.trajectory_patterns["visual_progression"] = []

        if perturbation_data["visual_impact"]["theme_change"]:
            self.trajectory_patterns["visual_progression"].append(
                {
                    "step": step_idx,
                    "app": perturbation_data["target_app"],
                    "theme": self.visual_state_tracker[perturbation_data["target_app"]]["current_theme"],
                }
            )

    def add_perturbation_result(
        self,
        step_idx: int,
        target_app: str,
        perturbation_type: str,
        command: str,
        success: bool,
        error_message: str = None,
    ):
        """Add perturbation result with success/failure tracking"""

        # Update the most recent perturbation with result
        for p in reversed(self.perturbation_history):
            if (
                p["step_idx"] == step_idx
                and p["target_app"] == target_app
                and p["perturbation_type"] == perturbation_type
            ):
                p["success"] = success
                p["error_message"] = error_message
                break

    def get_recent_perturbations(self, target_app: str = None, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent perturbations with enhanced context"""
        if target_app:
            app_perturbations = [p for p in self.perturbation_history if p["target_app"] == target_app]
            return app_perturbations[-limit:] if app_perturbations else []
        return self.perturbation_history[-limit:] if self.perturbation_history else []

    def get_coherence_context(self, target_app: str, current_step: int, task_progress: str = "middle") -> str:
        """DEPRECATED: Use get_context_for_decision instead"""
        context = self.get_context_for_decision(target_app, current_step, task_progress)
        return f"Context for {target_app} (Step {current_step}, Task: {task_progress}): {len(context['recent_perturbations'])} recent perturbations"

    def get_context_for_decision(
        self, target_app: str, current_step: int, task_progress: str = "middle"
    ) -> Dict[str, Any]:
        """Get comprehensive context for PerturbationGenerator decision-making"""
        recent = self.get_recent_perturbations(target_app, 5)
        failures = [p for p in recent if not p.get("success", True)]

        # Calculate context metrics
        failure_rate = len(failures) / max(len(recent), 1) if recent else 0
        repetition_count = len([p for p in recent if p["perturbation_type"] == "theme"])  # Default type

        # Visual state context
        visual_state = self.visual_state_tracker.get(target_app, {})
        current_theme = visual_state.get("current_theme", "default")
        recent_visual_changes = visual_state.get("visual_changes", [])

        # Generate contextual hints
        hints = self._generate_contextual_hints(target_app, recent, failures, task_progress)

        return {
            "recent_perturbations": recent,
            "failure_rate": failure_rate,
            "repetition_count": repetition_count,
            "current_visual_state": {
                "theme": current_theme,
                "recent_changes": recent_visual_changes[-3:] if recent_visual_changes else [],
            },
            "trajectory_patterns": {
                "perturbation_frequency": len(self.perturbation_history),
                "apps_affected": {p["target_app"] for p in self.perturbation_history},
                "visual_progression": self.trajectory_patterns.get("visual_progression", []),
            },
            "contextual_hints": hints,
            "task_progress": task_progress,
            "current_step": current_step,
        }

    def _generate_contextual_hints(
        self, target_app: str, recent: List[Dict], failures: List[Dict], task_progress: str
    ) -> List[str]:
        """Generate contextual hints for LLM decision-making"""
        hints = []

        # Failure pattern hints
        if len(failures) >= 2:
            hints.append(f"⚠️ High failure rate: {len(failures)} recent failures for {target_app}")

        # Repetition hints
        theme_perturbations = [p for p in recent if p["perturbation_type"] == "theme"]
        if len(theme_perturbations) >= 2:
            commands = [p["command"] for p in theme_perturbations]
            if len(set(commands)) == 1:
                hints.append(
                    f"⚠️ Exact command repetition: '{commands[0][:30]}...' used {len(theme_perturbations)} times"
                )
            else:
                hints.append(
                    f"ℹ️ Theme perturbations used {len(theme_perturbations)} times recently - consider variation"
                )

        # Visual state hints
        if target_app in self.visual_state_tracker:
            visual_state = self.visual_state_tracker[target_app]
            current_theme = visual_state.get("current_theme", "default")
            recent_changes = visual_state.get("visual_changes", [])

            if recent_changes:
                last_change = recent_changes[-1]
                hints.append(
                    f"🎨 Current theme: {current_theme}, last change: {last_change['type']} at step {last_change['step']}"
                )
            else:
                hints.append(f"🎨 Current theme: {current_theme}, no recent visual changes")

        # Learning opportunity hints
        if len(theme_perturbations) == 0:
            hints.append("💡 Good opportunity for theme change - no recent theme perturbations")
        elif target_app in ["chrome", "google-chrome", "chromium"]:
            hints.append("💡 CSS injection ideal for browser apps - high visual impact, low risk")

        # Task progress hints
        if task_progress == "beginning":
            hints.append("🚀 Task beginning - establish visual baseline with subtle changes")
        elif task_progress == "end":
            hints.append("🏁 Task ending - maintain visual consistency for completion")
        else:
            hints.append("⚡ Task middle - maximize learning with diverse perturbations")

        return hints

    def get_successful_alternatives(self, target_app: str, perturbation_type: str) -> List[str]:
        """Get successful alternative commands with visual coherence"""
        recent = self.get_recent_perturbations(target_app, 5)
        successful = [
            p for p in recent if p.get("success", True) and p["perturbation_type"] == perturbation_type
        ]

        # Prioritize alternatives that maintain visual coherence
        if target_app in self.visual_state_tracker:
            visual_state = self.visual_state_tracker[target_app]
            current_theme = visual_state["current_theme"]

            # Filter alternatives that match current visual state
            coherent_alternatives = []
            for p in successful:
                if perturbation_type == "theme":
                    if current_theme in p["command"].lower():
                        coherent_alternatives.append(p["command"])
                else:
                    coherent_alternatives.append(p["command"])

            return (
                coherent_alternatives[-3:]
                if coherent_alternatives
                else [p["command"] for p in successful[-3:]]
            )

        return [p["command"] for p in successful[-3:]]

    def get_trajectory_coherence_summary(self) -> Dict[str, Any]:
        """Get trajectory-level coherence summary"""
        return {
            "total_perturbations": len(self.perturbation_history),
            "apps_affected": {p["target_app"] for p in self.perturbation_history},
            "visual_states": self.visual_state_tracker,
            "perturbation_patterns": self.trajectory_patterns.get("perturbation_frequency", []),
            "visual_progression": self.trajectory_patterns.get("visual_progression", []),
        }


class OperationCatalog:
    """Comprehensive catalog of ALL available operations for perturbation generation"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
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
        """Load VLC-specific operations with concrete visual impact"""
        return {
            "playback": ["play", "pause", "stop", "next", "previous", "seek", "set_volume"],
            "settings": ["set_settings", "get_settings", "set_fullscreen", "set_theme"],
            "playlist": ["add_to_playlist", "remove_from_playlist", "clear_playlist", "get_playlist"],
            "ui": ["set_layout", "toggle_controls", "set_window_size", "set_zoom"],
            "media": ["load_media", "get_media_info", "set_audio_track", "set_subtitle_track"],
            "visual_effects": [
                "apply_video_filter_blur",
                "apply_video_filter_sepia",
                "apply_video_filter_invert",
                "change_aspect_ratio_4_3",
                "change_aspect_ratio_16_9",
                "change_aspect_ratio_stretch",
                "modify_video_brightness",
                "modify_video_contrast",
                "modify_video_saturation",
            ],
            "interface_customization": [
                "randomize_control_colors",
                "change_progress_bar_style",
                "modify_volume_slider_style",
                "alter_menu_transparency",
                "change_window_decoration",
                "modify_subtitle_styling",
            ],
            "system_integration": [
                "change_desktop_theme_for_vlc",
                "modify_system_fonts",
                "alter_cursor_theme",
                "change_wallpaper_background",
                "modify_window_manager_theme",
            ],
        }

    def _load_chrome_operations(self) -> Dict[str, List[str]]:
        """Load Chrome-specific operations with concrete visual impact"""
        return {
            "navigation": ["navigate", "go_back", "go_forward", "refresh", "reload"],
            "tabs": ["open_tab", "close_tab", "switch_tab", "new_tab", "get_tabs"],
            "bookmarks": ["bookmark_page", "get_bookmarks", "delete_bookmark", "create_bookmark_folder"],
            "settings": ["open_settings", "set_theme", "set_language", "set_homepage"],
            "devtools": ["open_devtools", "inspect_element", "console_log", "network_monitor"],
            "extensions": ["install_extension", "disable_extension", "get_extensions"],
            "security": ["clear_cookies", "clear_cache", "set_privacy_settings"],
            "visual_manipulation": [
                "inject_custom_css_red_theme",
                "inject_custom_css_dark_theme",
                "inject_custom_css_high_contrast",
                "modify_page_colors_scheme",
                "change_font_rendering_smooth",
                "alter_scrollbar_appearance_custom",
                "modify_focus_indicators_thick",
                "inject_css_blur_elements",
                "inject_css_rotate_elements",
            ],
            "content_perturbation": [
                "randomize_images_placeholders",
                "modify_text_content_obfuscate",
                "change_link_styling_colors",
                "alter_form_appearance_custom",
                "inject_fake_loading_bars",
                "modify_button_styles_rounded",
                "change_input_field_colors",
                "alter_table_border_styles",
                "modify_heading_font_sizes",
            ],
            "system_integration": [
                "change_desktop_theme_for_chrome",
                "modify_system_fonts",
                "alter_cursor_theme",
                "change_wallpaper_background",
                "modify_window_manager_theme",
            ],
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
        """Format operations for LLM prompt with validation and concrete examples"""
        if not window_states:
            self.logger.error("No window states provided to format_operations_for_llm")
            raise ValueError("No window states provided")

        normalized_states = normalize_window_states(window_states)
        formatted_operations = []

        # Validate operations exist in controller
        _validated_ops = self._validate_operations_against_controller()

        for window_state in normalized_states:
            app_name = window_state.app_name
            app_ops = self.get_operations_for_app(app_name)

            if app_ops:
                app_formatted = f"VALIDATED OPERATIONS FOR {app_name.upper()}:\n"
                for category, operations in app_ops.items():
                    # Filter to only include validated operations
                    validated_operations = [op for op in operations if self._is_operation_validated(op)]
                    if validated_operations:
                        app_formatted += f"  {category}: {', '.join(validated_operations)}\n"
                formatted_operations.append(app_formatted)
            else:
                # Include system operations if no app-specific operations found
                formatted_operations.append(
                    f"OPERATIONS FOR {app_name.upper()}:\n  system: Using validated system-level operations\n"
                )

        # Always include validated system operations as fallback
        system_ops = self.catalog["system_operations"]
        system_formatted = "VALIDATED SYSTEM OPERATIONS (guaranteed to work):\n"
        for category, operations in system_ops.items():
            validated_operations = [op for op in operations if self._is_operation_validated(op)]
            if validated_operations:
                system_formatted += f"  {category}: {', '.join(validated_operations)}\n"
        formatted_operations.append(system_formatted)

        # Include validated visual manipulation operations
        visual_ops = self.catalog["visual_manipulation"]
        visual_formatted = "VALIDATED VISUAL MANIPULATION OPERATIONS:\n"
        for category, operations in visual_ops.items():
            validated_operations = [op for op in operations if self._is_operation_validated(op)]
            if validated_operations:
                visual_formatted += f"  {category}: {', '.join(validated_operations)}\n"
        formatted_operations.append(visual_formatted)

        # Generate concrete examples
        examples = self._generate_concrete_examples_for_apps(normalized_states)
        formatted_operations.append(f"CONCRETE EXAMPLES:\n{examples}")

        # Add usage guidelines
        guidelines = self._generate_usage_guidelines()
        formatted_operations.append(f"USAGE GUIDELINES:\n{guidelines}")

        return "\n".join(formatted_operations)

    def _validate_operations_against_controller(self) -> Dict[str, List[str]]:
        """Validate that operations are actually implemented in the controller"""
        # This is a simplified validation - in practice, you'd check against actual controller methods
        validated_operations = {
            "execute_bash_command": True,
            "execute_python_command": True,
            "execute_js_on_page": True,
            "execute_uno_command": True,
            "execute_css_injection": True,
            "execute_dom_modification": True,
            "execute_theme_randomization": True,
            "execute_layout_perturbation": True,
            "execute_typography_randomization": True,
            "execute_animation_effects": True,
            "execute_accessibility_perturbation": True,
            "execute_system_perturbation": True,
            "change_wallpaper": True,
            "set_theme": True,
            "set_font_size": True,
            "set_window_size": True,
            "toggle_sidebar": True,
            "format_range": True,
            "set_color": True,
            "set_font": True,
        }
        return validated_operations

    def _is_operation_validated(self, operation: str) -> bool:
        """Check if an operation is validated against controller implementation"""
        validated_ops = self._validate_operations_against_controller()
        return operation in validated_ops

    def _generate_concrete_examples_for_apps(self, window_states: List[Any]) -> str:
        """Generate concrete examples for the specific apps in window states"""
        app_names = [ws.app_name for ws in window_states]
        examples = OperationExamples.get_all_examples_for_apps(app_names)
        return "\n".join([f"- {example}" for example in examples[:20]])  # Limit to 20 examples

    def _generate_usage_guidelines(self) -> str:
        """Generate usage guidelines for operations"""
        guidelines = [
            "1. Always use concrete commands with specific parameters",
            "2. Test operations in the target environment before suggesting",
            "3. Provide fallback options for robustness",
            "4. Use system-level operations as safe fallbacks",
            "5. Ensure commands are feasible for Ubuntu environment",
            "6. Prefer operations that maintain application functionality",
            "7. Include error handling in generated commands",
            "8. Use validated operations only - avoid experimental features",
        ]
        return "\n".join(guidelines)


class BaseLLM:
    """
    Base LLM interface supporting multiple providers

    Supported model formats:
    - Gemini: "gemini-2.5-flash-lite", "gemini-2.5-flash", etc.
    - OpenAI: "gpt-5-nano", "gpt-4o-mini", "gpt-5-mini", "gpt-5", "gpt-4o", etc.
    - Anthropic: "claude-haiku-3.5", "claude-sonnet-4", etc.
    - OpenRouter: "openrouter-glm-4.5-air", "openrouter-claude-3.5-sonnet", etc.

    Environment variables required:
    - GEMINI_API_KEY for Gemini models
    - OPENAI_API_KEY for OpenAI models
    - ANTHROPIC_API_KEY for Anthropic models
    - OPENROUTER_API_KEY for OpenRouter models
    """

    def __init__(self, model_name: str = "gemini-2.5-flash-lite", model_provider: str = "gemini"):
        self.model_provider = model_provider
        self.model_name = model_name

        # TODO: Change provider and model
        # self.model_provider = "openai"
        # self.model_name = "gpt-4o-mini"

        # self.model_provider = "anthropic"
        # self.model_name = "claude-haiku-3.5"

        # self.model_provider = "openrouter"
        # self.model_name = "z-ai/glm-4.5-air:free"

        self.logger = logging.getLogger(__name__)

        # Initialize app-specific strategies (shared by all LLM classes)
        self.app_specific_strategies = self._build_app_specific_strategies()

        # Initialize client based on model type
        self.client = None
        self.api_key = None

        if self.model_provider == "gemini":
            self.api_key = os.getenv("GEMINI_API_KEY")
            if self.api_key:
                self.client = genai.Client()
            else:
                self.logger.warning("Gemini API not available - using mock responses")
        elif self.model_provider == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY")
            if self.api_key:
                from openai import OpenAI

                self.client = OpenAI(api_key=self.api_key)
            else:
                self.logger.warning("OpenAI API not available - using mock responses")
        elif self.model_provider == "anthropic":
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
            if self.api_key:
                from anthropic import Anthropic

                self.client = Anthropic(api_key=self.api_key)
            else:
                self.logger.warning("Anthropic API not available - using mock responses")
        elif self.model_provider == "openrouter":
            self.api_key = os.getenv("OPENROUTER_API_KEY")
            if self.api_key:
                from openai import OpenAI

                self.client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=self.api_key,
                )
            else:
                self.logger.warning("OpenRouter API not available - using mock responses")
        else:
            self.logger.warning(f"Unknown model type: {self.model_name} - using mock responses")

    def call_llm(self, prompt: str) -> str:
        """Call LLM with prompt using the correct API format for each provider"""
        if not self.client:
            return '{"error": "Mock response - API not available"}'

        retries = 0
        max_retries = 3
        while retries < max_retries:
            try:
                retries += 1

                if self.model_provider == "gemini":
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            thinking_config=types.ThinkingConfig(thinking_budget=0)
                        ),
                    )
                    if response and hasattr(response, "text") and response.text:
                        return response.text
                    else:
                        raise ValueError("No response text from Gemini API")

                elif self.model_provider == "openai":
                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,  # Low temperature for consistent results
                        max_tokens=4000,  # Reasonable limit for JSON responses
                    )
                    if completion.choices and len(completion.choices) > 0:
                        return completion.choices[0].message.content
                    else:
                        raise ValueError("No response content from OpenAI API")

                elif self.model_provider == "anthropic":
                    message = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=4000,
                        temperature=0.1,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    if message.content and len(message.content) > 0:
                        return message.content[0].text
                    else:
                        raise ValueError("No response content from Anthropic API")

                elif self.model_provider == "openrouter":
                    # OpenRouter API format (uses OpenAI-compatible interface)
                    completion = self.client.chat.completions.create(
                        extra_headers={},
                        extra_body={},
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    if completion.choices and len(completion.choices) > 0:
                        return completion.choices[0].message.content
                    else:
                        raise ValueError("No response content from OpenRouter API")

            except Exception as e:
                self.logger.error(f"Error calling LLM: {e}, retrying {retries}/{max_retries}...")

        return f"{'error: LLM call failed after {max_retries} attempts'}"

    def extract_json(self, response: str) -> List[Dict[str, Any]]:
        """Extract JSON with multiple fallback strategies"""
        try:
            # Strategy 1: Standard JSON block extraction
            json_str = self._extract_json_block(response)
            if json_str:
                parsed = json.loads(json_str)
                return [parsed] if isinstance(parsed, dict) else parsed

            # Strategy 2: Try to find JSON-like structures
            json_str = self._extract_json_patterns(response)
            if json_str:
                parsed = json.loads(json_str)
                return [parsed] if isinstance(parsed, dict) else parsed

            # Strategy 3: Generate minimal valid response
            self.logger.warning("JSON extraction failed, using fallback response")
            return [self._generate_fallback_response()]

        except Exception as e:
            self.logger.error(f"JSON extraction failed: {e}")
            return [self._generate_fallback_response()]

    def _extract_json_block(self, response: str) -> Optional[str]:
        """Extract JSON from code blocks"""
        try:
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                if json_end > json_start:
                    return response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                if json_end > json_start:
                    return response[json_start:json_end].strip()
            else:
                # Try to find JSON in the response
                response_stripped = response.strip()
                if response_stripped.startswith("{") or response_stripped.startswith("["):
                    return response_stripped
            return None
        except Exception:
            return None

    def _extract_json_patterns(self, response: str) -> Optional[str]:
        """Extract JSON using pattern matching"""
        import re

        # Look for JSON-like structures
        json_patterns = [
            r'\{[^{}]*"[^"]*"[^{}]*\}',  # Simple object
            r"\[[^\[\]]*\{[^}]*\}[^\[\]]*\]",  # Array of objects
            r"\{.*?\}",  # Any object-like structure
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                try:
                    # Validate it's actually JSON
                    json.loads(match)
                    return match
                except Exception:
                    continue
        return None

    def _generate_fallback_response(self, target_app: str = "system") -> Dict[str, Any]:
        """Generate minimal valid perturbation decision that avoids harmful operations"""
        return {
            "should_apply": False,
            "reasoning": "LLM response parsing failed - using safe fallback that avoids harmful operations",
            "perturbation_type": "theme",
            "api_call": "execute_bash_command",
            "generated_command": 'echo "Safe fallback - no visual changes applied"',
            "parameters": {"target_app": target_app, "intensity": "low"},
            "confidence": 0.1,
            "alternative_commands": [],
            "visual_impact": "No visual changes - safe fallback",
            "coherence_rationale": "Fallback response to prevent harmful operations",
        }

    def _format_app_states_for_decision(self, app_states: List[Any]) -> str:
        """Format app states for perturbation decision"""
        if not app_states:
            return "No app states available"

        normalized_states = normalize_window_states(app_states)
        formatted_states = []

        for window_state in normalized_states:
            app_summary = self._format_single_window_state(window_state, normalized_states)
            formatted_states.append(app_summary)

        return "\n".join(formatted_states)

    def _format_single_window_state(
        self, window_state: WindowState, all_window_states: List[WindowState]
    ) -> str:
        """Format a single window state with hierarchical element tree"""
        app_name = window_state.app_name
        window_name = window_state.window_name

        # Show window hierarchy information with better context
        if window_name != app_name:
            app_summary = f"App: {app_name} - Window: {window_name}\n"
        else:
            app_summary = f"App: {app_name}\n"

        # Add context about whether this is browser chrome or webpage content
        if app_name.lower() in ["chrome", "chromium", "google-chrome"]:
            # Check if this window contains webpage content vs browser chrome
            if "chrome://" in window_name.lower() or "chrome-" in window_name.lower():
                app_summary += "  [BROWSER CHROME - NOT WEBPAGE CONTENT]\n"
            else:
                app_summary += "  [WEBPAGE CONTENT - PRIORITIZE THESE ELEMENTS]\n"

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
                all_window_states=all_window_states,
            )
        else:
            app_summary += "  (no elements detected)\n"

        return app_summary

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

        # Show ALL elements in the tree
        # Add context hints based on element properties
        context_hints = self._get_element_context_hints(element, window_state, all_window_states)

        # Format element name and type
        display_name = element.name if element.name else f"{element.element_type}"

        # Format position
        pos = element.position
        position_str = f"at ({pos.get('center_x', 0)}, {pos.get('center_y', 0)})" if pos else "no position"

        # Add element line
        result += f"{indent}- {display_name} ({element.element_type}){context_hints} {position_str}\n"

        # Traverse all children
        for child in element.children:
            child_result = self._format_element_tree_hierarchical(
                child, depth + 1, window_state, all_window_states
            )
            result += child_result

        return result

    def _is_element_blocked_by_z_order(
        self, element: UIElement, window_state: WindowState, all_window_states: List[WindowState]
    ) -> bool:
        """Check if element is blocked by a higher z-order window"""
        from perturbation_engine.tools.app_state_manager import ElementValidator

        # Use the centralized ElementValidator from app_state_manager
        validator = ElementValidator()
        return validator._is_element_blocked_by_z_order(element, all_window_states)

    def _get_element_context_hints(
        self,
        element: UIElement,
        window_state: WindowState = None,
        all_window_states: List[WindowState] = None,
    ) -> str:
        """Get context hints for element based on its properties and z-order blocking"""
        hints = []

        # Browser chrome detection hints
        if window_state and window_state.app_name.lower() in ["chrome", "chromium", "google-chrome"]:
            # Check if this is likely a browser chrome element vs webpage element
            element_name_lower = (element.name or "").lower()
            if any(
                chrome_indicator in element_name_lower
                for chrome_indicator in [
                    "address",
                    "omnibox",
                    "bookmark",
                    "toolbar",
                    "menu",
                    "tab",
                    "chrome",
                    "browser",
                ]
            ):
                hints.append("[BROWSER CHROME]")
            elif element_name_lower in ["search", "shop", "google"]:
                hints.append("[WEBPAGE ELEMENT]")

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

        # Visibility hints
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

    def _build_app_specific_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Build app-specific perturbation strategies for effective invariant feature learning"""
        # Base strategies to avoid duplication
        browser_strategy = {
            "primary_focus": "CSS injection, DOM manipulation, visual styling",
            "invariant_learning_goals": [
                "Learn to recognize UI elements across different color schemes",
                "Adapt to varying button styles, borders, and layouts",
                "Handle dynamic content changes and visual overlays",
                "Recognize elements with different fonts and typography",
            ],
            "perturbation_types": [
                "css_injection",
                "dom_modification",
                "visual_randomization",
                "theme",
                "typography",
                "layout",
            ],
            "safe_operations": [
                "execute_css_injection",
                "execute_dom_modification",
                "execute_theme_randomization",
                "execute_layout_perturbation",
            ],
            "avoid_operations": ["execute_bash_command", "execute_uno_command"],
            "multi_app_strategy": "Use background browser windows with different themes as visual interference",
        }

        libreoffice_strategy = {
            "primary_focus": "Theme changes, formatting, data visualization without data corruption",
            "invariant_learning_goals": [
                "Recognize elements across different themes and color schemes",
                "Adapt to varying formatting and layout styles",
                "Handle different toolbar appearances and menu layouts",
                "Recognize elements with different visual presentations",
            ],
            "perturbation_types": ["theme", "visual_formatting", "system_level", "layout"],
            "safe_operations": [
                "execute_system_theme_coherence",
                "execute_libreoffice_visual_formatting",
                "execute_bash_command",
            ],
            "avoid_operations": ["execute_css_injection", "execute_dom_modification"],
            "content_safety": "Never modify actual content, only visual presentation",
            "multi_app_strategy": "Use background LibreOffice windows with different themes",
        }

        editor_strategy = {
            "primary_focus": "Editor themes, syntax highlighting, interface customization",
            "invariant_learning_goals": [
                "Recognize code elements across different syntax highlighting themes",
                "Adapt to varying editor layouts and panel arrangements",
                "Handle different font rendering and typography",
                "Recognize UI elements with different visual styles",
            ],
            "perturbation_types": ["theme", "typography", "layout", "system_level", "visual_randomization"],
            "safe_operations": [
                "execute_python_command",
                "execute_system_theme_coherence",
                "execute_bash_command",
            ],
            "avoid_operations": ["execute_css_injection", "execute_dom_modification", "execute_uno_command"],
        }

        vlc_strategy = {
            "primary_focus": "Media player themes, visual effects, interface customization",
            "invariant_learning_goals": [
                "Recognize media controls across different themes",
                "Adapt to varying player interface layouts",
                "Handle different control button styles and positions",
                "Recognize media elements with different visual presentations",
            ],
            "perturbation_types": ["theme", "visual_effects", "system_level", "layout", "gui_manipulation"],
            "safe_operations": [
                "execute_vlc_visual_effects",
                "execute_system_theme_coherence",
                "execute_python_command",
            ],
            "avoid_operations": ["execute_css_injection", "execute_dom_modification", "execute_uno_command"],
        }

        multi_app_strategy = {
            "primary_focus": "Cross-app visual interference, background processes, system-level changes",
            "invariant_learning_goals": [
                "Learn to focus on target app despite visual distractions",
                "Adapt to varying system-wide theme changes",
                "Handle multiple app windows with different appearances",
                "Recognize target elements despite background interference",
            ],
            "perturbation_types": [
                "system_level",
                "background_process",
                "window_management",
                "cross_app_interference",
                "notification",
            ],
            "safe_operations": [
                "execute_bash_command",
                "execute_system_integration",
                "execute_background_app_launch",
            ],
            "strategy": "Launch background apps/windows that don't block target elements",
            "safety_notes": "Ensure background apps don't overlap with target UI elements",
        }

        return {
            # Browser variants
            "chrome": {
                **browser_strategy,
                "concrete_examples": [
                    "execute_css_injection('body { background-color: #f0f8ff !important; }', {'target_app': 'chrome'})",
                    "execute_css_injection('button { border-radius: 12px !important; box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important; }', {'target_app': 'chrome'})",
                    "execute_dom_modification('document.querySelectorAll(\"button\").forEach(btn => btn.style.filter = \"hue-rotate(90deg)\")', {'target_app': 'chrome'})",
                ],
            },
            "google-chrome": {
                **browser_strategy,
                "concrete_examples": [
                    "execute_css_injection('body { background-color: #f0f8ff !important; }', {'target_app': 'chrome'})",
                    "execute_css_injection('button { border-radius: 12px !important; box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important; }', {'target_app': 'chrome'})",
                    "execute_dom_modification('document.querySelectorAll(\"button\").forEach(btn => btn.style.filter = \"hue-rotate(90deg)\")', {'target_app': 'chrome'})",
                ],
            },
            "chromium": {
                **browser_strategy,
                "concrete_examples": [
                    "execute_css_injection('body { background-color: #f0f8ff !important; }', {'target_app': 'chromium'})",
                    "execute_css_injection('button { border-radius: 12px !important; box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important; }', {'target_app': 'chromium'})",
                    "execute_dom_modification('document.querySelectorAll(\"button\").forEach(btn => btn.style.filter = \"hue-rotate(90deg)\")', {'target_app': 'chromium'})",
                ],
            },
            # LibreOffice variants
            "libreoffice_calc": {
                **libreoffice_strategy,
                "concrete_examples": [
                    "execute_bash_command('gsettings set org.gnome.desktop.interface gtk-theme \"Adwaita-dark\"')",
                    "execute_libreoffice_visual_formatting('randomize_toolbar_colors', {'target_app': 'calc'})",
                    "execute_libreoffice_visual_formatting('change_grid_line_style', {'target_app': 'calc'})",
                ],
            },
            "calc": {
                **libreoffice_strategy,
                "concrete_examples": [
                    "execute_bash_command('gsettings set org.gnome.desktop.interface gtk-theme \"Adwaita-dark\"')",
                    "execute_libreoffice_visual_formatting('randomize_toolbar_colors', {'target_app': 'calc'})",
                    "execute_libreoffice_visual_formatting('change_grid_line_style', {'target_app': 'calc'})",
                ],
            },
            "libreoffice_writer": {
                **libreoffice_strategy,
                "concrete_examples": [
                    "execute_bash_command('gsettings set org.gnome.desktop.interface gtk-theme \"Adwaita\"')",
                    "execute_libreoffice_visual_formatting('change_font_rendering', {'target_app': 'writer'})",
                    "execute_libreoffice_visual_formatting('randomize_toolbar_appearance', {'target_app': 'writer'})",
                ],
            },
            "writer": {
                **libreoffice_strategy,
                "concrete_examples": [
                    "execute_bash_command('gsettings set org.gnome.desktop.interface gtk-theme \"Adwaita\"')",
                    "execute_libreoffice_visual_formatting('change_font_rendering', {'target_app': 'writer'})",
                    "execute_libreoffice_visual_formatting('randomize_toolbar_appearance', {'target_app': 'writer'})",
                ],
            },
            "libreoffice_impress": {
                **libreoffice_strategy,
                "concrete_examples": [
                    "execute_bash_command('gsettings set org.gnome.desktop.interface gtk-theme \"Adwaita-dark\"')",
                    "execute_libreoffice_visual_formatting('randomize_slide_backgrounds', {'target_app': 'impress'})",
                    "execute_libreoffice_visual_formatting('change_toolbar_appearance', {'target_app': 'impress'})",
                ],
            },
            "impress": {
                **libreoffice_strategy,
                "concrete_examples": [
                    "execute_bash_command('gsettings set org.gnome.desktop.interface gtk-theme \"Adwaita-dark\"')",
                    "execute_libreoffice_visual_formatting('randomize_slide_backgrounds', {'target_app': 'impress'})",
                    "execute_libreoffice_visual_formatting('change_toolbar_appearance', {'target_app': 'impress'})",
                ],
            },
            # Editor variants
            "code": {
                **editor_strategy,
                "concrete_examples": [
                    "execute_python_command('CodeTools.set_theme(\"dark\")')",
                    "execute_python_command('CodeTools.set_font_size(16)')",
                    "execute_python_command('CodeTools.toggle_sidebar()')",
                ],
            },
            "vscode": {
                **editor_strategy,
                "concrete_examples": [
                    "execute_python_command('CodeTools.set_theme(\"dark\")')",
                    "execute_python_command('CodeTools.set_font_size(16)')",
                    "execute_python_command('CodeTools.toggle_sidebar()')",
                ],
            },
            # Media player
            "vlc": {
                **vlc_strategy,
                "concrete_examples": [
                    'execute_python_command(\'VLCTools.set_settings("qt-theme", "dark")\')',
                    "execute_vlc_visual_effects('apply_video_filter_sepia', {'target_app': 'vlc'})",
                    "execute_vlc_visual_effects('change_aspect_ratio_16_9', {'target_app': 'vlc'})",
                ],
            },
            # Multi-app
            "multi_app": {
                **multi_app_strategy,
                "concrete_examples": [
                    "execute_bash_command('gnome-terminal --title=\"Background Terminal\" &')",
                    'execute_bash_command(\'notify-send "Multi-App Test" "Background notification"\')',
                    "execute_bash_command('gsettings set org.gnome.desktop.interface gtk-theme \"Yaru-dark\"')",
                    "execute_system_integration({'action': 'launch_background_app', 'app': 'calculator'})",
                ],
            },
        }


class CurriculumGenerator(BaseLLM):
    """LLM-driven curriculum generator for diverse and strategic perturbation scenarios"""

    def __init__(self, model_name: str = "gemini-2.0-flash-lite", model_provider: str = "gemini"):
        super().__init__(model_name, model_provider)
        self.operation_catalog = OperationCatalog()
        self.verifier = LLMOutputVerifier()

    def generate_scenario_specs(
        self, seed_trajectory: SeedTrajectory, window_states: List[Any], curriculum_config: CurriculumConfig
    ) -> List[ScenarioSpec]:
        """Generate diverse and strategic scenario specifications using LLM with full operation awareness"""
        task_context = self._analyze_task_context_with_llm(seed_trajectory, window_states, curriculum_config)

        # Use scenario_count from curriculum_config (1-100+ based on configuration)
        target_scenario_count = curriculum_config.scenario_count
        self.logger.info(
            f"Generating {target_scenario_count} diverse scenarios for task {seed_trajectory.task_id}"
        )

        scenarios = self._generate_diverse_scenarios_with_llm(task_context, target_scenario_count)
        validated_scenarios = self._validate_scenarios(scenarios, task_context, seed_trajectory.task_id)
        diverse_scenarios = self._ensure_curriculum_diversity(validated_scenarios, task_context)
        prioritized_scenarios = self._prioritize_scenarios(diverse_scenarios, task_context)

        return prioritized_scenarios[:target_scenario_count]

    def _generate_diverse_scenarios_with_llm(
        self, task_context: Dict[str, Any], target_count: int
    ) -> List[Dict[str, Any]]:
        """Generate diverse scenarios using systematic diversity enforcement (1-100+ based on target_count)"""
        all_scenarios = []
        max_retries = 3

        # Dynamically adjust batch size based on target count
        if target_count <= 10:
            batch_size = 5  # Small batches for small counts
        elif target_count <= 50:
            batch_size = 10  # Medium batches for medium counts
        else:
            batch_size = 20  # Large batches for large counts (100+)

        num_batches = (target_count + batch_size - 1) // batch_size
        self.logger.info(f"Generating {target_count} scenarios in {num_batches} batches of {batch_size}")

        for batch_idx in range(num_batches):
            batch_scenarios = []
            remaining_count = min(batch_size, target_count - len(all_scenarios))

            # Store current batch size for diversity constraints
            self._current_batch_size = batch_size

            for attempt in range(max_retries):
                try:
                    # Create batch-specific diversity constraints
                    diversity_constraints = self._create_batch_diversity_constraints(
                        batch_idx, num_batches, all_scenarios, task_context
                    )

                    prompt = self._create_diverse_curriculum_prompt_with_constraints(
                        task_context, remaining_count, diversity_constraints
                    )
                    response = self.call_llm(prompt)
                    scenarios_data = self.extract_json(response)

                    if scenarios_data:
                        batch_scenarios = scenarios_data
                        break
                    else:
                        if attempt < max_retries - 1:
                            self.logger.warning(
                                f"Batch {batch_idx + 1} attempt {attempt + 1} failed: No scenarios generated, retrying..."
                            )
                            continue
                        else:
                            self.logger.error(
                                f"Batch {batch_idx + 1} failed after 3 attempts: No scenarios generated"
                            )
                            # Generate fallback scenarios for this batch
                            batch_scenarios = self._generate_fallback_scenarios(
                                remaining_count, task_context, batch_idx
                            )
                            break

                except Exception as e:
                    if attempt < max_retries - 1:
                        self.logger.warning(
                            f"Batch {batch_idx + 1} attempt {attempt + 1} failed: {e}, retrying..."
                        )
                        continue
                    else:
                        self.logger.error(f"Batch {batch_idx + 1} failed after 3 attempts: {e}")
                        # Generate fallback scenarios for this batch
                        batch_scenarios = self._generate_fallback_scenarios(
                            remaining_count, task_context, batch_idx
                        )
                        break

            all_scenarios.extend(batch_scenarios)
            self.logger.info(
                f"Generated batch {batch_idx + 1}/{num_batches}: {len(batch_scenarios)} scenarios"
            )

        return all_scenarios[:target_count]

    def _create_batch_diversity_constraints(
        self,
        batch_idx: int,
        total_batches: int,
        existing_scenarios: List[Dict[str, Any]],
        task_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create batch-specific diversity constraints to ensure 100 unique scenarios"""
        # Analyze existing scenarios to avoid duplicates
        used_combinations = set()
        used_apps = set()
        used_types = set()
        used_intensities = set()
        used_triggers = set()

        for scenario in existing_scenarios:
            app = scenario.get("target_app", "system")
            types = tuple(sorted(scenario.get("perturbation_types", [])))
            intensity = scenario.get("perturbation_intensity", "medium")
            trigger = scenario.get("perturbation_trigger", "")

            used_combinations.add((app, types, intensity))
            used_apps.add(app)
            used_types.update(types)
            used_intensities.add(intensity)
            used_triggers.add(trigger)

        # Define comprehensive diversity dimensions for systematic coverage
        target_apps = task_context.get("app_types", ["system"])

        # Realistic GUI perturbation types for invariant learning
        perturbation_types = [
            # Core visual appearance changes (realistic GUI variations)
            "theme",
            "color",
            "typography",
            "layout",
            "shape",
            "density",
            # Realistic GUI environmental conditions
            "display_scaling",
            "screen_resolution",
            "system_theme",
            "accessibility_mode",
            # Content and data changes (safe visual only)
            "content_variation",
            "data_visualization",
            "ui_state",
            "accessibility",
            # UI structure changes (realistic web/desktop variations)
            "ui_injection",
            "dom_modification",
            "css_injection",
            "visual_randomization",
            # System and environment changes (realistic system variations)
            "system_level",
            "background_process",
            "window_management",
            "file_operations",
            # Cross-app interference (realistic multi-app scenarios)
            "notification",
            "cross_app_interference",
            "gui_manipulation",
            # Realistic GUI state variations
            "ui_state",
            "window_state",
            "content_state",
            "loading_state",
            "animation_state",
            # App-specific realistic variations
            "playback",
            "settings",
            "playlist",
            "navigation",
            "tabs",
            "bookmarks",
            "devtools",
            "file_ops",
            "editing",
            # Realistic visual effects
            "animation_effects",
            "transition_effects",
            "hover_effects",
            "focus_effects",
            # Accessibility and usability (realistic accessibility scenarios)
            "accessibility_perturbation",
            "keyboard_navigation",
            "screen_reader",
        ]

        # Expanded intensity levels
        intensities = ["very_low", "low", "medium", "high", "very_high"]

        # Realistic GUI trigger conditions
        triggers = [
            # Realistic app lifecycle scenarios
            "During app startup",
            "When app window is focused",
            "When app window loses focus",
            "When switching between app windows",
            "When app is minimized/maximized",
            # Realistic user interaction scenarios
            "When user hovers over elements",
            "When user clicks on elements",
            "When user types in input fields",
            "When user navigates between pages",
            "When user opens/closes menus",
            "When user scrolls through content",
            # Realistic system event scenarios
            "When system theme changes",
            "When display scaling changes",
            "When system notifications appear",
            "When background apps update",
            "When network connectivity changes",
            "When system resources are low",
            # Realistic content loading scenarios
            "When page content is loading",
            "When images are loading",
            "When videos are buffering",
            "When data is being processed",
            "When files are being saved",
            "When documents are being opened",
            # Realistic accessibility scenarios
            "When high contrast mode is enabled",
            "When screen reader is active",
            "When keyboard navigation is used",
            "When text scaling is increased",
            # Realistic multi-app scenarios
            "When multiple apps are running",
            "When switching between browser tabs",
            "When background apps show notifications",
            "When system dialogs appear",
        ]

        # Additional diversity dimensions
        timing_options = ["initial", "runtime", "between_steps", "after_completion", "on_error"]
        target_scopes = ["system", "app", "window", "element", "content", "background"]
        visual_impact_levels = ["subtle", "moderate", "dramatic", "extreme"]
        learning_focus_areas = [
            "color_invariance",
            "layout_invariance",
            "typography_invariance",
            "shape_invariance",
            "size_invariance",
            "position_invariance",
            "theme_invariance",
            "accessibility_invariance",
            "interaction_invariance",
        ]

        # Calculate batch-specific constraints (dynamic based on actual batch size)
        # Get the actual batch size from the calling context
        scenarios_per_batch = getattr(self, "_current_batch_size", 20)  # Default to 20 if not set
        start_idx = batch_idx * scenarios_per_batch

        return {
            "batch_index": batch_idx,
            "total_batches": total_batches,
            "used_combinations": list(used_combinations),
            "used_apps": list(used_apps),
            "used_types": list(used_types),
            "used_intensities": list(used_intensities),
            "used_triggers": list(used_triggers),
            "target_apps": target_apps,
            "available_types": perturbation_types,
            "available_intensities": intensities,
            "available_triggers": triggers,
            "timing_options": timing_options,
            "target_scopes": target_scopes,
            "visual_impact_levels": visual_impact_levels,
            "learning_focus_areas": learning_focus_areas,
            "scenario_start_index": start_idx,
            "diversity_requirements": {
                "min_new_apps": max(1, len(target_apps) // total_batches),
                "min_new_types": max(5, len(perturbation_types) // total_batches),
                "min_new_intensities": max(2, len(intensities) // total_batches),
                "min_new_triggers": max(4, len(triggers) // total_batches),
                "min_new_timing": max(1, len(timing_options) // total_batches),
                "min_new_scopes": max(2, len(target_scopes) // total_batches),
                "min_new_visual_impacts": max(1, len(visual_impact_levels) // total_batches),
                "min_new_learning_focus": max(2, len(learning_focus_areas) // total_batches),
            },
        }

    def _create_diverse_curriculum_prompt_with_constraints(
        self, task_context: Dict[str, Any], scenario_count: int, diversity_constraints: Dict[str, Any]
    ) -> str:
        """Create curriculum prompt with batch-specific diversity constraints"""
        # Use the existing prompt creation method as base
        base_prompt = self._create_diverse_curriculum_prompt(task_context, scenario_count)

        # Add batch-specific diversity constraints
        batch_constraints = f"""
BATCH-SPECIFIC DIVERSITY CONSTRAINTS (Batch {diversity_constraints["batch_index"] + 1}/{diversity_constraints["total_batches"]}):

AVOID THESE ALREADY USED COMBINATIONS:
- Used Apps: {", ".join(diversity_constraints["used_apps"])}
- Used Types: {", ".join(diversity_constraints["used_types"])}
- Used Intensities: {", ".join(diversity_constraints["used_intensities"])}
- Used Triggers: {", ".join(diversity_constraints["used_triggers"][:10])}...

MANDATORY DIVERSITY REQUIREMENTS FOR THIS BATCH:
- Use at least {diversity_constraints["diversity_requirements"]["min_new_apps"]} different target apps
- Use at least {diversity_constraints["diversity_requirements"]["min_new_types"]} different perturbation types
- Use at least {diversity_constraints["diversity_requirements"]["min_new_intensities"]} different intensities
- Use at least {diversity_constraints["diversity_requirements"]["min_new_triggers"]} different triggers

AVAILABLE DIVERSITY DIMENSIONS:
- Target Apps: {", ".join(diversity_constraints["target_apps"])}
- Perturbation Types: {", ".join(diversity_constraints["available_types"])}
- Intensities: {", ".join(diversity_constraints["available_intensities"])}
- Triggers: {", ".join(diversity_constraints["available_triggers"])}

SCENARIO INDEXING:
- This batch covers scenarios {diversity_constraints["scenario_start_index"]} to {diversity_constraints["scenario_start_index"] + scenario_count - 1}
- Each scenario must be uniquely identifiable and different from all previous scenarios

CRITICAL: Each scenario in this batch must be completely different from all previously generated scenarios.
"""

        return base_prompt + batch_constraints

    def _generate_fallback_scenarios(
        self, count: int, task_context: Dict[str, Any], batch_idx: int
    ) -> List[Dict[str, Any]]:
        """Generate fallback scenarios when LLM generation fails"""
        fallback_scenarios = []
        target_apps = task_context.get("app_types", ["system"])

        # Create systematic fallback scenarios
        base_scenarios = [
            {
                "target_app": "system",
                "perturbation_trigger": "During task execution",
                "available_perturbation_actions": 'execute_bash_command(\'notify-send "System Notification" "Visual change applied"\')',
                "learning_objectives": "Learn to handle system-level visual changes",
                "target_components": ["system_notifications"],
                "perturbation_types": ["notification"],
                "perturbation_category": "system_level",
                "perturbation_intensity": "low",
                "maintains_functionality": True,
                "maintains_accessibility": True,
                "realistic_scenario": "System notification appears during task execution",
                "initial_state_perturbation": False,
                "runtime_perturbation": True,
                "risk_mitigation": "Non-blocking notification that doesn't interfere with UI elements",
                "educational_rationale": "Tests ability to ignore non-critical system notifications",
            }
        ]

        # Generate variations
        batch_size = getattr(self, "_current_batch_size", 20)  # Get current batch size
        for i in range(count):
            scenario = base_scenarios[i % len(base_scenarios)].copy()
            scenario["target_app"] = target_apps[i % len(target_apps)]
            scenario["perturbation_trigger"] = f"During step {i + 1} of task execution"
            scenario["learning_objectives"] = (
                f"Learn visual invariance for scenario {batch_idx * batch_size + i + 1}"
            )
            fallback_scenarios.append(scenario)

        return fallback_scenarios

    def _create_diverse_curriculum_prompt(self, task_context: Dict[str, Any], scenario_count: int) -> str:
        """Create curriculum prompt with concrete, feasible operations"""
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

        # Generate concrete operation examples based on target apps
        target_apps = task_context.get("app_types", ["system"])
        concrete_examples = self._generate_concrete_operation_examples(target_apps)

        # Generate app-specific guidance
        app_specific_guidance = self._generate_app_specific_guidance(target_apps)

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

CONCRETE OPERATION EXAMPLES FOR TARGET APPS:
{concrete_examples}

APP-SPECIFIC GUIDANCE FOR INVARIANT FEATURE LEARNING:
{app_specific_guidance}

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
    "available_perturbation_actions": "concrete_executable_command_with_specific_parameters",
    "learning_objectives": "specific_learning_goal_for_visual_invariance",
    "target_components": ["specific_ui_elements_to_target"],
    "perturbation_types": ["theme", "layout", "content_variation"],
    "perturbation_category": "app_specific",
    "perturbation_intensity": "low",
    "maintains_functionality": true,
    "maintains_accessibility": true,
    "realistic_scenario": "brief_explanation_of_realistic_context",
    "initial_state_perturbation": true,
    "runtime_perturbation": true,
    "risk_mitigation": "brief_explanation_of_safety_measures",
    "educational_rationale": "brief_explanation_of_learning_value"
}}

CRITICAL REQUIREMENTS:
1. Each scenario must be UNIQUE and cover different aspects of visual invariance learning
2. available_perturbation_actions must be concrete, executable commands with specific parameters
3. Use ONLY validated API calls: execute_bash_command, execute_python_command, execute_css_injection, execute_uno_command
4. Commands MUST be syntactically correct and tested for Ubuntu environment
5. Focus on perturbations that maintain target element accessibility
6. Ensure commands are feasible for the target application
7. Prefer simple, reliable commands over complex ones
8. Include specific parameters and values that work in Ubuntu environment

DIVERSITY ENFORCEMENT:
- NO two scenarios should use the same perturbation type + intensity combination
- NO two scenarios should target the same app with the same trigger condition
- Each scenario must teach a DIFFERENT aspect of visual invariance
- Mix visual-only changes with functional-safe changes
- Vary timing: some initial, some runtime, some between-steps
- Include both app-specific and system-level perturbations

SAFETY REQUIREMENTS:
- NEVER include commands that could delete files, close apps, or corrupt data
- ONLY use commands that change visual appearance safely
- Ensure commands maintain application functionality
- Test each command mentally for safety before including

QUALITY VALIDATION:
- Test each command mentally before including it
- Ensure commands are complete and executable
- Use concrete examples from the operation catalog
- Focus on meaningful visual changes for learning objectives
"""
        return prompt

    def _generate_concrete_operation_examples(self, target_apps: List[str]) -> str:
        """Generate concrete operation examples for target apps efficiently"""
        examples = []

        for app in target_apps:
            app_lower = app.lower()
            app_strategy = self.app_specific_strategies.get(app_lower)

            if app_strategy and "concrete_examples" in app_strategy:
                examples.extend(app_strategy["concrete_examples"])
            else:
                examples.extend(OperationExamples.get_examples_for_app(app))

        # Add system examples
        examples.extend(OperationExamples.get_system_examples())

        return "\n".join(f"- {example}" for example in examples[:20])  # Reduced limit

    def _generate_app_specific_guidance(self, target_apps: List[str]) -> str:
        """Generate app-specific guidance efficiently"""
        guidance_parts = []

        for app in target_apps:
            app_strategy = self.app_specific_strategies.get(app.lower())
            if not app_strategy:
                continue

            guidance_parts.append(f"\n{app.upper()} GUIDANCE:")

            if "primary_focus" in app_strategy:
                guidance_parts.append(f"Focus: {app_strategy['primary_focus']}")

            if "safe_operations" in app_strategy:
                guidance_parts.append(f"Safe: {', '.join(app_strategy['safe_operations'][:2])}")

            if "avoid_operations" in app_strategy:
                guidance_parts.append(f"Avoid: {', '.join(app_strategy['avoid_operations'])}")

        return "\n".join(guidance_parts)

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
        triggers_used = set()
        apps_used = set()

        for scenario in scenarios:
            categories_used.add(scenario.perturbation_category.value)
            types_used.update([pt.value for pt in scenario.perturbation_types])
            intensities_used.add(scenario.perturbation_intensity.value)
            triggers_used.add(scenario.perturbation_trigger)
            apps_used.add(scenario.target_app)

        # Log diversity analysis
        self.logger.info("Curriculum diversity analysis:")
        self.logger.info(f"  Categories covered: {len(categories_used)}/{len(PerturbationCategory)}")
        self.logger.info(f"  Types covered: {len(types_used)}/{len(PerturbationType)}")
        self.logger.info(f"  Intensities covered: {len(intensities_used)}/{len(PerturbationIntensity)}")
        self.logger.info(f"  Unique triggers: {len(triggers_used)}")
        self.logger.info(f"  Unique apps: {len(apps_used)}")

        # Enforce diversity requirements
        min_diversity_threshold = 0.6  # At least 60% of available diversity
        min_categories = max(2, int(len(PerturbationCategory) * min_diversity_threshold))
        min_types = max(3, int(len(PerturbationType) * min_diversity_threshold))
        min_intensities = max(2, int(len(PerturbationIntensity) * min_diversity_threshold))

        diversity_score = (
            len(categories_used) / min_categories
            + len(types_used) / min_types
            + len(intensities_used) / min_intensities
            + len(triggers_used) / len(scenarios)
            + len(apps_used) / len(scenarios)
        ) / 5

        self.logger.info(f"  Diversity score: {diversity_score:.2f} (target: 1.0)")

        if diversity_score < 0.8:
            self.logger.warning(
                f"Low diversity detected ({diversity_score:.2f}). Consider regenerating scenarios."
            )

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


class PerturbationGenerator(BaseLLM):
    """LLM-driven perturbation generator with comprehensive operation awareness and procedural memory"""

    def __init__(self, model_name: str = "gemini-2.0-flash-lite", model_provider: str = "gemini"):
        super().__init__(model_name, model_provider)
        self.operation_catalog = OperationCatalog()
        self.procedural_memory = ProceduralMemory()
        self.verifier = LLMOutputVerifier()

        # Safety patterns for harmful commands
        self.harmful_patterns = {
            "file_deletion": ["rm ", "delete", "remove", "unlink", "trash"],
            "app_closing": ["close", "quit", "exit", "kill", "terminate"],
            "data_corruption": ["corrupt", "damage", "destroy", "overwrite", "truncate"],
            "ui_blocking": ["hide", "disable", "block", "obscure", "cover"],
            "system_dangerous": ["sudo", "chmod 777", "chown", "format", "fdisk"],
        }

        # Safe perturbation patterns
        self.safe_patterns = {
            "visual_only": ["color", "theme", "font", "background", "border", "opacity"],
            "layout_safe": ["margin", "padding", "spacing", "size", "position"],
            "content_safe": ["text", "label", "placeholder", "tooltip", "hint"],
        }

        # App-specific perturbation strategies for invariant feature learning
        self.app_specific_strategies = self._build_app_specific_strategies()

    def decide_perturbation(
        self, execution_context: ExecutionContext, scenario_spec: ScenarioSpec
    ) -> Dict[str, Any]:
        """Decide whether to apply perturbation with rich procedural memory context"""

        # Get LLM decision with enhanced procedural memory context
        llm_decision = self._get_llm_decision_with_context(execution_context, scenario_spec)

        if llm_decision.get("should_apply", False):
            # Validate command safety before applying
            generated_command = llm_decision.get("generated_command", "")
            target_app = scenario_spec.target_app

            is_safe, safety_reason = self._validate_command_safety(
                generated_command, target_app, execution_context.window_states
            )

            if not is_safe:
                llm_decision["should_apply"] = False
                llm_decision["reasoning"] = f"Command safety check failed: {safety_reason}"
                llm_decision["error_message"] = safety_reason
                return llm_decision

            # Update procedural memory with the decision (only if safe)
            self._update_procedural_memory(execution_context, scenario_spec, llm_decision)

            llm_decision["procedural_memory_enhanced"] = True
            llm_decision["reasoning"] += " (Enhanced with procedural memory context)"

        return llm_decision

    def _validate_command_safety(
        self, command: str, target_app: str, window_states: List[Any]
    ) -> Tuple[bool, str]:
        """Validate that a command is safe and won't break the trajectory"""
        if not command:
            return False, "Empty command"

        command_lower = command.lower()

        # Check for harmful patterns
        for category, patterns in self.harmful_patterns.items():
            for pattern in patterns:
                if pattern in command_lower:
                    return False, f"Command contains harmful pattern '{pattern}' (category: {category})"

        # Check for safe patterns
        has_safe_pattern = False
        for _category, patterns in self.safe_patterns.items():
            for pattern in patterns:
                if pattern in command_lower:
                    has_safe_pattern = True
                    break

        if not has_safe_pattern:
            return False, "Command doesn't contain safe perturbation patterns"

        # Enhanced safety checks
        safety_checks = [
            self._check_data_integrity_protection(command_lower),
            self._check_element_blocking_prevention(command_lower, target_app, window_states),
            self._check_file_system_safety(command_lower),
            self._check_application_stability(command_lower, target_app),
            self._check_ui_accessibility_maintenance(command_lower, target_app),
        ]

        for is_safe, reason in safety_checks:
            if not is_safe:
                return False, reason

        return True, "Command appears safe"

    def _check_data_integrity_protection(self, command_lower: str) -> Tuple[bool, str]:
        """Check if command protects data integrity"""
        # Commands that could corrupt or modify data
        data_corruption_patterns = [
            "rm ",
            "delete",
            "remove",
            "unlink",
            "trash",
            "mv ",
            "cp ",
            "dd ",
            "overwrite",
            "truncate",
            "corrupt",
            "damage",
            "destroy",
            "wipe",
            "format",
            "fdisk",
            "mkfs",
            "chmod 777",
            "chown",
            "chgrp",
        ]

        for pattern in data_corruption_patterns:
            if pattern in command_lower:
                return False, f"Command could corrupt data: contains '{pattern}'"

        return True, "Data integrity protected"

    def _check_element_blocking_prevention(
        self, command_lower: str, target_app: str, window_states: List[Any]
    ) -> Tuple[bool, str]:
        """Check if command would block target elements"""
        # Commands that could hide or block UI elements
        blocking_patterns = [
            "hide",
            "disable",
            "block",
            "obscure",
            "cover",
            "overlay",
            "z-index",
            "position: fixed",
            "display: none",
            "visibility: hidden",
            "opacity: 0",
            "width: 0",
            "height: 0",
            "minimize",
            "close",
        ]

        for pattern in blocking_patterns:
            if pattern in command_lower:
                return False, f"Command could block elements: contains '{pattern}'"

        # Check for window management commands that could affect target app
        if self._would_block_target_app(command_lower, target_app, window_states):
            return False, "Command would block or hide target application"

        return True, "Element accessibility maintained"

    def _check_file_system_safety(self, command_lower: str) -> Tuple[bool, str]:
        """Check if command is safe for file system operations"""
        # Dangerous file system operations
        dangerous_fs_patterns = [
            "sudo",
            "su ",
            "chmod 777",
            "chown root",
            "chgrp root",
            "rm -rf",
            "rm -r",
            "rm -f",
            "dd if=",
            "mkfs",
            "fdisk",
            "format",
            "wipe",
            "shred",
            "secure-delete",
        ]

        for pattern in dangerous_fs_patterns:
            if pattern in command_lower:
                return False, f"Command is dangerous for file system: contains '{pattern}'"

        return True, "File system operations safe"

    def _check_application_stability(self, command_lower: str, target_app: str) -> Tuple[bool, str]:
        """Check if command maintains application stability"""
        # Commands that could crash or destabilize applications
        stability_patterns = [
            "kill",
            "terminate",
            "pkill",
            "killall",
            "xkill",
            "crash",
            "abort",
            "exit",
            "quit",
            "close",
        ]

        for pattern in stability_patterns:
            if pattern in command_lower and target_app.lower() in command_lower:
                return False, f"Command could destabilize {target_app}: contains '{pattern}'"

        return True, "Application stability maintained"

    def _check_ui_accessibility_maintenance(self, command_lower: str, target_app: str) -> Tuple[bool, str]:
        """Check if command maintains UI accessibility"""
        # Commands that could break accessibility
        accessibility_breaking_patterns = [
            "aria-hidden",
            "tabindex=-1",
            "disabled",
            "readonly",
            "pointer-events: none",
            "user-select: none",
        ]

        for pattern in accessibility_breaking_patterns:
            if pattern in command_lower:
                return False, f"Command could break accessibility: contains '{pattern}'"

        return True, "UI accessibility maintained"

    def _validate_element_reachability(
        self, target_element: Any, window_states: List[Any], perturbation_command: str = None
    ) -> Tuple[bool, str]:
        """Validate that target elements remain reachable after perturbation"""
        from perturbation_engine.tools.app_state_manager import ElementValidator

        # Use the centralized ElementValidator from app_state_manager
        validator = ElementValidator()
        return validator.validate_element_reachability(target_element, window_states, perturbation_command)

    def _create_data_integrity_protection_system(self) -> Dict[str, Any]:
        """Create comprehensive data integrity protection system"""
        return {
            "protected_file_patterns": [
                "*.docx",
                "*.xlsx",
                "*.pptx",
                "*.pdf",
                "*.txt",
                "*.csv",
                "*.json",
                "*.xml",
                "*.html",
                "*.css",
                "*.js",
                "*.py",
                "*.cpp",
                "*.java",
                "*.c",
                "*.h",
                "*.hpp",
            ],
            "protected_directory_patterns": [
                "/home/*/Documents",
                "/home/*/Desktop",
                "/home/*/Downloads",
                "/opt/*",
                "/usr/*",
                "/etc/*",
                "/var/*",
                "/tmp/*/important",
            ],
            "safe_visual_operations": [
                "theme_change",
                "color_modification",
                "font_change",
                "layout_adjustment",
                "background_change",
                "border_modification",
                "opacity_adjustment",
                "animation_effects",
                "transition_effects",
                "hover_effects",
            ],
            "data_safe_commands": [
                "notify-send",
                "gsettings set org.gnome.desktop.interface",
                "gsettings set org.gnome.desktop.background",
                "echo",
                "printf",
                "css_injection",
                "dom_modification",
                "theme_randomization",
            ],
            "backup_requirements": {
                "create_backup_before_modification": True,
                "backup_location": "/tmp/perturbation_backups",
                "backup_retention_days": 7,
                "verify_backup_integrity": True,
            },
        }

    def _validate_data_integrity_protection(
        self, command: str, target_app: str, window_states: List[Any]
    ) -> Tuple[bool, str]:
        """Validate that command maintains data integrity"""
        protection_system = self._create_data_integrity_protection_system()
        command_lower = command.lower()

        # Check if command is in safe operations list
        is_safe_operation = False
        for safe_op in protection_system["safe_visual_operations"]:
            if safe_op.replace("_", " ") in command_lower:
                is_safe_operation = True
                break

        # Check if command uses data-safe commands
        is_data_safe_command = False
        for safe_cmd in protection_system["data_safe_commands"]:
            if safe_cmd in command_lower:
                is_data_safe_command = True
                break

        if not is_safe_operation and not is_data_safe_command:
            return False, "Command is not in approved data-safe operations list"

        # Check for file system operations that could affect data
        if self._would_affect_protected_files(command_lower, protection_system):
            return False, "Command could affect protected files or directories"

        return True, "Data integrity protection maintained"

    def _would_affect_protected_files(self, command_lower: str, protection_system: Dict[str, Any]) -> bool:
        """Check if command would affect protected files or directories"""
        # Check for file operations
        file_operation_patterns = [
            "> ",
            ">> ",
            "< ",
            "<< ",
            "cat ",
            "head ",
            "tail ",
            "grep ",
            "sed ",
            "awk ",
            "find ",
            "locate ",
            "which ",
            "whereis ",
        ]

        has_file_operation = False
        for pattern in file_operation_patterns:
            if pattern in command_lower:
                has_file_operation = True
                break

        if not has_file_operation:
            return False  # No file operations detected

        # Check if command targets protected patterns
        for protected_pattern in protection_system["protected_file_patterns"]:
            if protected_pattern.replace("*", "") in command_lower:
                return True

        for protected_dir in protection_system["protected_directory_patterns"]:
            if protected_dir.replace("*", "").replace("/", " ") in command_lower:
                return True

        return False

    def _would_block_target_app(self, command: str, target_app: str, window_states: List[Any]) -> bool:
        """Check if command would block or hide the target application"""
        command_lower = command.lower()

        # Check for commands that might close or hide the target app
        blocking_patterns = [
            f"close {target_app.lower()}",
            f"hide {target_app.lower()}",
            f"minimize {target_app.lower()}",
            f"kill {target_app.lower()}",
            f"quit {target_app.lower()}",
            f"exit {target_app.lower()}",
        ]

        for pattern in blocking_patterns:
            if pattern in command_lower:
                return True

        # Check for system-level changes that might affect the app
        if "gsettings" in command_lower and "gtk-theme" in command_lower:
            # Theme changes can cause LibreOffice to show dialogs unexpectedly
            return True

        return False

    def _update_procedural_memory(
        self, execution_context: ExecutionContext, scenario_spec: ScenarioSpec, llm_decision: Dict[str, Any]
    ):
        """Update procedural memory with the perturbation decision"""
        try:
            target_app = scenario_spec.target_app.lower()
            perturbation_type = llm_decision.get("perturbation_type", "theme")
            generated_command = llm_decision.get("generated_command", "")

            if generated_command:
                self.procedural_memory.add_perturbation(
                    execution_context.step_idx,
                    target_app,
                    perturbation_type,
                    generated_command,
                    execution_context.window_states[0] if execution_context.window_states else {},
                )
        except Exception as e:
            self.logger.error(f"Error updating procedural memory: {e}")

    def _determine_task_progress(self, step_idx: int, task_instruction: str, total_steps: int = None) -> str:
        """Determine task progress based on step index and task complexity"""
        if total_steps is None:
            # Estimate total steps based on task complexity
            total_steps = self._estimate_task_complexity(task_instruction)

        progress_ratio = step_idx / total_steps if total_steps > 0 else 0

        if progress_ratio <= 0.2:
            return "beginning"
        elif progress_ratio >= 0.8:
            return "end"
        else:
            return "middle"

    def _estimate_task_complexity(self, task_instruction: str) -> int:
        """Estimate task complexity based on instruction content"""
        instruction_lower = task_instruction.lower()

        # Simple heuristics for task complexity
        complexity_indicators = {
            "simple": ["click", "open", "close", "save"],
            "medium": ["create", "edit", "format", "navigate", "search"],
            "complex": ["analyze", "calculate", "pivot", "macro", "script", "automate"],
        }

        _simple_count = sum(
            1 for indicator in complexity_indicators["simple"] if indicator in instruction_lower
        )
        medium_count = sum(
            1 for indicator in complexity_indicators["medium"] if indicator in instruction_lower
        )
        complex_count = sum(
            1 for indicator in complexity_indicators["complex"] if indicator in instruction_lower
        )

        # Estimate steps based on complexity
        if complex_count > 0:
            return 20  # Complex tasks: 15-25 steps
        elif medium_count > 0:
            return 12  # Medium tasks: 8-15 steps
        else:
            return 8  # Simple tasks: 5-10 steps

    def _detect_multi_app_scenario(self, execution_context: ExecutionContext) -> bool:
        """Detect if this is a multi-app scenario based on task config or window states"""
        # Check task config for multi-app tags
        if hasattr(execution_context, "scenario_spec") and execution_context.scenario_spec:
            scenario_spec = execution_context.scenario_spec
            if hasattr(scenario_spec, "target_app"):
                target_app = scenario_spec.target_app.lower()
                if "multi" in target_app or "multiapp" in target_app:
                    return True

        # Check window states for multiple active apps
        if execution_context.window_states:
            active_apps = set()
            for window_state in execution_context.window_states:
                if hasattr(window_state, "is_active") and window_state.is_active:
                    app_name = window_state.app_name.lower()
                    if app_name not in ["gnome-shell", "gjs", "desktop"]:
                        active_apps.add(app_name)

            # Multi-app if more than one non-desktop app is active
            return len(active_apps) > 1

        return False

    def _get_app_specific_strategy(
        self, target_app: str, execution_context: ExecutionContext
    ) -> Dict[str, Any]:
        """Get app-specific perturbation strategy based on target app and context"""
        # Check for multi-app scenario first
        if self._detect_multi_app_scenario(execution_context):
            return self.app_specific_strategies.get("multi_app", {})

        # Direct lookup - strategies are already mapped by app name
        app_lower = target_app.lower()
        strategy = self.app_specific_strategies.get(app_lower)

        # Fallback to browser strategy if no specific strategy found
        if not strategy:
            strategy = self.app_specific_strategies.get("chrome", {})
            self.logger.warning(f"No specific strategy found for {target_app}, using browser strategy")

        return strategy

    def _build_app_specific_context(self, app_strategy: Dict[str, Any], target_app: str) -> str:
        """Build app-specific context for LLM prompt efficiently"""
        if not app_strategy:
            return f"No specific strategy available for {target_app}"

        # Build context parts efficiently
        context_parts = []

        # Essential fields only
        if "primary_focus" in app_strategy:
            context_parts.append(f"Primary Focus: {app_strategy['primary_focus']}")

        if "invariant_learning_goals" in app_strategy:
            context_parts.append("Key Learning Goals:")
            context_parts.extend(f"  - {goal}" for goal in app_strategy["invariant_learning_goals"][:3])

        if "safe_operations" in app_strategy:
            context_parts.append(f"Safe Operations: {', '.join(app_strategy['safe_operations'][:3])}")

        if "avoid_operations" in app_strategy:
            context_parts.append(f"Avoid: {', '.join(app_strategy['avoid_operations'])}")

        return "\n".join(context_parts)

    def _format_operations_for_prompt(self, app_operations: Dict[str, List[str]]) -> str:
        """Format operations for LLM prompt efficiently"""
        if not app_operations:
            return "No specific operations available"

        formatted_parts = []
        for category, operations in app_operations.items():
            if operations:
                formatted_parts.append(f"{category.upper()}:")
                for operation in operations[:5]:  # Limit to 5 operations per category
                    formatted_parts.append(f"  - {operation}")

        return "\n".join(formatted_parts) if formatted_parts else "No operations available"

    def _format_procedural_memory_context(self, procedural_context: Dict[str, Any]) -> str:
        """Format procedural memory context for LLM prompt"""
        context_parts = []

        # Task progress context
        task_progress = procedural_context.get("task_progress", "middle")
        current_step = procedural_context.get("current_step", 0)
        context_parts.append(f"TASK PROGRESS: {task_progress.upper()} (Step {current_step})")

        # Perturbation history context
        recent_perturbations = procedural_context.get("recent_perturbations", [])
        if recent_perturbations:
            context_parts.append("RECENT PERTURBATIONS:")
            for i, p in enumerate(recent_perturbations[-3:], 1):
                success_indicator = "✓" if p.get("success", True) else "✗"
                context_parts.append(
                    f"  {i}. {success_indicator} Step {p['step_idx']}: {p['perturbation_type']} - {p['command'][:50]}..."
                )
        else:
            context_parts.append("RECENT PERTURBATIONS: None")

        # Visual state context
        visual_state = procedural_context.get("current_visual_state", {})
        if visual_state:
            theme = visual_state.get("theme", "default")
            recent_changes = visual_state.get("recent_changes", [])
            context_parts.append(f"CURRENT VISUAL STATE: Theme={theme}")
            if recent_changes:
                context_parts.append("RECENT VISUAL CHANGES:")
                for change in recent_changes[-2:]:
                    context_parts.append(f"  - {change['type']} at step {change['step']}")

        # Contextual hints
        hints = procedural_context.get("contextual_hints", [])
        if hints:
            context_parts.append("CONTEXTUAL HINTS:")
            for hint in hints:
                context_parts.append(f"  {hint}")

        # Trajectory patterns
        trajectory_patterns = procedural_context.get("trajectory_patterns", {})
        if trajectory_patterns:
            total_perturbations = trajectory_patterns.get("perturbation_frequency", 0)
            apps_affected = trajectory_patterns.get("apps_affected", [])
            context_parts.append(
                f"TRAJECTORY PATTERNS: {total_perturbations} total perturbations, apps: {', '.join(apps_affected)}"
            )

        return "\n".join(context_parts)

    def _get_llm_decision_with_context(
        self, execution_context: ExecutionContext, scenario_spec: ScenarioSpec
    ) -> Dict[str, Any]:
        """Get LLM decision with procedural memory context and retries"""
        try:
            # Determine task progress based on step index
            task_progress = self._determine_task_progress(
                execution_context.step_idx, execution_context.task_instruction, execution_context.total_steps
            )

            # Get rich procedural memory context
            procedural_context = self.procedural_memory.get_context_for_decision(
                scenario_spec.target_app, execution_context.step_idx, task_progress
            )

            # Format procedural memory context for prompt
            memory_context = self._format_procedural_memory_context(procedural_context)

            # Get app-specific strategy
            app_strategy = self._get_app_specific_strategy(scenario_spec.target_app, execution_context)

            # Get available operations for target app (optimized)
            app_operations = self.operation_catalog.get_operations_for_app(scenario_spec.target_app.lower())
            formatted_operations = self._format_operations_for_prompt(app_operations)

            # Build app-specific context
            app_context = self._build_app_specific_context(app_strategy, scenario_spec.target_app)

            prompt = f"""
{PROMPT_CONSTANTS["perturbation_role"]}

CURRENT EXECUTION CONTEXT:
Step: {execution_context.step_idx}
Next Action: {execution_context.current_action}
Task: {execution_context.task_instruction}
App States: {self._format_app_states_for_decision(execution_context.window_states)}

SCENARIO SPECIFICATION:
Target App: {scenario_spec.target_app}
Trigger: {scenario_spec.perturbation_trigger}
Available Actions: {scenario_spec.available_perturbation_actions}
Learning Objectives: {scenario_spec.learning_objectives}
Perturbation Types: {[pt.value for pt in scenario_spec.perturbation_types]}

APP-SPECIFIC STRATEGY FOR {scenario_spec.target_app.upper()}:
{app_context}

AVAILABLE OPERATIONS FOR {scenario_spec.target_app.upper()}:
{formatted_operations}

PROCEDURAL MEMORY CONTEXT:
{memory_context}

COHERENCE REQUIREMENTS:
1. Build on previous successful perturbations in the trajectory
2. Create meaningful visual impact for learning objectives
3. Maintain application functionality and accessibility
4. Use concrete, executable commands with specific parameters
5. Consider app-specific visual perturbation opportunities

PERTURBATION DECISION CRITERIA:
{chr(10).join([f"{i + 1}. {criteria}" for i, criteria in enumerate(PROMPT_CONSTANTS["perturbation_criteria"])])}

DECISION EXAMPLES:
{chr(10).join([f"- {example}" for example in PROMPT_CONSTANTS["decision_examples"]])}

SAFETY REQUIREMENTS (CRITICAL):
1. NEVER use commands that could:
   - Delete files or data (rm, delete, remove, unlink, trash)
   - Close or hide applications (close, quit, exit, kill, terminate)
   - Corrupt or damage data (corrupt, damage, destroy, overwrite)
   - Block UI elements (hide, disable, block, obscure, cover)
   - Use dangerous system commands (sudo, chmod 777, format, fdisk)

2. ONLY use commands that:
   - Change visual appearance (colors, themes, fonts, backgrounds)
   - Modify layout properties (margins, padding, spacing, sizes)
   - Update content safely (text, labels, placeholders, tooltips)
   - Maintain application functionality and accessibility

3. AVOID commands that might cause LibreOffice to show unexpected dialogs:
   - System theme changes (gsettings gtk-theme) can trigger file dialogs
   - UNO commands that modify internal LibreOffice state

COMMAND GENERATION RULES:
1. Use ONLY validated API calls: execute_bash_command, execute_python_command, execute_css_injection, execute_uno_command
2. Commands MUST be syntactically correct and executable
3. Include specific parameters and values that work in Ubuntu environment
4. Test commands mentally before suggesting them
5. Prefer simple, reliable commands over complex ones
6. Ensure commands maintain target element accessibility

SAFE COMMAND EXAMPLES:
- Chrome: execute_css_injection('body {{ background-color: #f0f0f0 !important; }}', {{'target_app': 'chrome'}})
- Chrome: execute_css_injection('button {{ border-radius: 8px !important; }}', {{'target_app': 'chrome'}})
- System: execute_bash_command('notify-send "Test notification" "Visual change applied"')
- LibreOffice: execute_css_injection('.toolbar {{ background-color: #e8e8e8 !important; }}', {{'target_app': 'libreoffice'}})

QUALITY REQUIREMENTS:
- generated_command must be a complete, executable command
- api_call must match one of the validated API calls
- parameters must include target_app and other required fields
- reasoning must explain why this perturbation is meaningful and coherent
- Command must NOT break the next action's feasibility

Return JSON:
{{
    "should_apply": true/false,
    "reasoning": "detailed_explanation_of_decision_with_coherence_considerations",
    "perturbation_type": "{"|".join([pt.value for pt in PerturbationType])}",
    "api_call": "execute_bash_command|execute_python_command|execute_css_injection|execute_uno_command",
    "generated_command": "complete_executable_command_with_specific_parameters",
    "parameters": {{
        "target_app": "{scenario_spec.target_app}",
        "intensity": "low|medium|high",
        "coherent_with_history": true/false,
        "maintains_functionality": true/false,
        "preserves_target_accessibility": true/false
    }},
    "confidence": 0.0-1.0,
    "alternative_commands": ["list_of_alternative_executable_commands"],
    "visual_impact": "description_of_expected_visual_changes",
    "coherence_rationale": "explanation_of_how_this_builds_on_previous_perturbations"
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
                return self._generate_fallback_response(scenario_spec.target_app)

            return result

        except Exception as e:
            self.logger.error(f"Error in _get_llm_decision_with_context: {e}")
            return self._create_error_fallback_response(scenario_spec.target_app, str(e))

    def _create_error_fallback_response(self, target_app: str, error_message: str) -> Dict[str, Any]:
        """Create a safe fallback response when errors occur"""
        return {
            "should_apply": False,
            "reasoning": f"Error occurred: {error_message}",
            "perturbation_type": "theme",
            "api_call": "execute_bash_command",
            "generated_command": 'echo "Safe fallback - no perturbation applied"',
            "parameters": {"target_app": target_app, "intensity": "low"},
            "confidence": 0.0,
            "alternative_commands": [],
            "visual_impact": "No visual changes - safe fallback",
            "coherence_rationale": "Fallback response due to error",
        }


class ElementIdentificationLLM(BaseLLM):
    """Element identification using LLM"""

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

        window_states_summary = self._format_app_states_for_decision(window_states)

        prompt = f"""
        Find ALL possible UI elements that this action could be trying to interact with.
        Return them ranked by likelihood, with the most likely first.

        Action: "{action_str}"

        Available Elements:
        {window_states_summary}

        CRITICAL PRIORITIZATION RULES:
        1. WEBPAGE CONTENT OVER BROWSER CHROME: Always prioritize elements marked as "[WEBPAGE CONTENT - PRIORITIZE THESE ELEMENTS]" over elements in "[BROWSER CHROME - NOT WEBPAGE CONTENT]"
        2. CONTEXT RELEVANCE: For actions mentioning "search bar", "page", "website", or specific webpage content, prioritize webpage elements
        3. ELEMENT TYPE APPROPRIATENESS: Match element types to action types (e.g., "CLICK" actions should prioritize clickable elements)
        4. VISIBILITY AND INTERACTIVITY: Prioritize visible, enabled, and interactive elements
        5. COORDINATE REASONABLENESS: Elements with coordinates (0,0) or outside screen bounds (0-1920, 0-1080) should be deprioritized

        DISAMBIGUATION RULES:
        - If multiple elements have the same name, consider ALL of them as potential candidates
        - Elements marked as "[collapsed menu - likely invisible]" or "[hidden dropdown - likely invisible]" should be deprioritized
        - Elements marked as "[inactive tab - likely invisible]" should be deprioritized
        - Elements marked as "[blocked by higher window]" should be deprioritized
        - Elements marked as "[disabled]" should be deprioritized
        - Very small elements (≤16x16 pixels) without names should be deprioritized unless they have interactive properties

        RANKING PRIORITY (in order):
        1. Webpage content elements (marked with "[WEBPAGE CONTENT - PRIORITIZE THESE ELEMENTS]")
        2. Element type appropriateness for the action
        3. Coordinate reasonableness and visibility
        4. Hierarchy visibility and context relevance
        5. Element naming and interactive properties

        Return JSON array with element identifiers, ranked by likelihood from 0.00 to 1.00 and only return elements with confidence values greater than 0.70:
        [
            {{
                "name": "element_name",
                "element_type": "element_type",
                "app_name": "app_name",
                "confidence": 0.95,
                "reasoning": "detailed_reasoning_for_element_selection_including_webpage_priority_and_coordinate_analysis"
            }},
            {{
                "name": "alternative_element_name",
                "element_type": "element_type",
                "app_name": "app_name",
                "confidence": 0.75,
                "reasoning": "alternative_reasoning_with_webpage_context_considerations"
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

    # def _format_window_states_for_llm(self, window_states: List[WindowState]) -> str:
    #     """Format window states with complete hierarchical element tree for LLM consumption"""
    #     return self._format_app_states_for_decision(window_states)


class QualityLLM(BaseLLM):
    """Quality evaluation"""

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
    """Simple LLM output verification"""

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
            "perturbation_types",
        ]

        for field in required_fields:
            if field not in scenario_data or not scenario_data[field]:
                errors.append(f"Missing required field: {field}")

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

        return len(errors) == 0, errors

    def verify_perturbation_decision(self, decision_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Verify perturbation decision completeness and validity"""
        errors = []

        required_fields = ["should_apply", "reasoning", "perturbation_type", "api_call"]

        for field in required_fields:
            if field not in decision_data:
                errors.append(f"Missing required field: {field}")

        # Validate should_apply
        if "should_apply" in decision_data:
            should_apply = decision_data["should_apply"]
            if not isinstance(should_apply, bool):
                errors.append("Invalid should_apply: must be boolean")

        return len(errors) == 0, errors

    def sanitize_scenario_data(self, scenario_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize and normalize scenario data"""
        sanitized = {}

        # Sanitize string fields
        string_fields = [
            "target_app",
            "perturbation_trigger",
            "available_perturbation_actions",
            "learning_objectives",
        ]

        for field in string_fields:
            if field in scenario_data:
                value = scenario_data[field]
                if isinstance(value, str):
                    sanitized[field] = value.strip()
                else:
                    sanitized[field] = str(value).strip()

        # Sanitize list fields
        if "perturbation_types" in scenario_data:
            value = scenario_data["perturbation_types"]
            if isinstance(value, list):
                sanitized["perturbation_types"] = [str(item).strip() for item in value if item]
            else:
                sanitized["perturbation_types"] = [str(value).strip()]

        return sanitized
