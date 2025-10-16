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

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from pydantic import BaseModel

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


# Pydantic models for structured output
class PerturbationOpportunity(BaseModel):
    element_type: str
    perturbation_category: str
    perturbation_type: str
    target_scope: str
    timing: str
    intensity: str
    educational_value: str
    risk_level: str
    maintains_accessibility: bool


class TaskCharacteristics(BaseModel):
    estimated_steps: str
    primary_apps: List[str]
    critical_elements: List[str]
    workflow_type: str


class TaskAnalysis(BaseModel):
    complexity: str
    domain: str
    learning_objectives: List[str]
    task_characteristics: TaskCharacteristics
    perturbation_opportunities: List[PerturbationOpportunity]
    reasoning: str


class ScenarioSpecForLLM(BaseModel):
    """Pydantic model for LLM structured output - converts to dataclass ScenarioSpec"""

    target_app: str
    perturbation_trigger: str
    available_perturbation_actions: str
    learning_objectives: str
    target_components: List[str]
    perturbation_types: List[str]
    perturbation_category: str
    perturbation_intensity: str = "medium"
    maintains_functionality: bool = True
    maintains_accessibility: bool = True
    realistic_scenario: str = ""
    initial_state_perturbation: bool = False
    runtime_perturbation: bool = True
    risk_mitigation: str = ""
    educational_rationale: str = ""

    def to_scenario_spec(self, task_id: str, scenario_index: int) -> ScenarioSpec:
        """Convert to the main dataclass ScenarioSpec with proper ID generation"""
        from perturbation_engine.pipeline.data_models import (
            PerturbationCategory,
            PerturbationIntensity,
            PerturbationType,
        )

        # Generate meaningful scenario ID: task_id + scenario_number + target_app
        scenario_id = f"{task_id}_scenario_{scenario_index + 1}_{self.target_app}"

        return ScenarioSpec(
            scenario_id=scenario_id,
            target_app=self.target_app,
            perturbation_trigger=self.perturbation_trigger,
            available_perturbation_actions=self.available_perturbation_actions,
            learning_objectives=self.learning_objectives,
            target_components=self.target_components,
            perturbation_types=[PerturbationType.from_string(pt) for pt in self.perturbation_types],
            perturbation_category=PerturbationCategory.from_string(self.perturbation_category),
            perturbation_intensity=PerturbationIntensity.from_string(self.perturbation_intensity),
            maintains_functionality=self.maintains_functionality,
            maintains_accessibility=self.maintains_accessibility,
            realistic_scenario=self.realistic_scenario,
            initial_state_perturbation=self.initial_state_perturbation,
            runtime_perturbation=self.runtime_perturbation,
            risk_mitigation=self.risk_mitigation,
            educational_rationale=self.educational_rationale,
        )


class ElementCandidate(BaseModel):
    name: str
    element_type: str
    app_name: str
    confidence: float
    reasoning: str


class PerturbationParameters(BaseModel):
    target_app: str
    intensity: str
    coherent_with_history: bool
    maintains_functionality: bool
    preserves_target_accessibility: bool


class PerturbationDecision(BaseModel):
    should_apply: bool
    reasoning: str
    perturbation_type: str
    api_call: str
    generated_command: str
    parameters: PerturbationParameters
    confidence: float
    alternative_commands: List[str]
    visual_impact: str
    coherence_rationale: str


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

        # Enhanced diversity analysis
        diversity_analysis = self._analyze_perturbation_diversity(recent, target_app)

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
            "diversity_analysis": diversity_analysis,
            "contextual_hints": hints,
            "task_progress": task_progress,
            "current_step": current_step,
        }

    def _analyze_perturbation_diversity(
        self, recent_perturbations: List[Dict], target_app: str
    ) -> Dict[str, Any]:
        """Analyze the diversity of recent perturbations to guide future decisions"""
        if not recent_perturbations:
            return {
                "diversity_score": 1.0,
                "missing_dimensions": [],
                "overused_dimensions": [],
                "recommendations": ["No recent perturbations - good opportunity for any type"],
            }

        # Analyze different diversity dimensions
        visual_intents = set()
        element_targets = set()
        perturbation_types = set()
        api_calls = set()

        for p in recent_perturbations:
            command = p.get("command", "")
            if command:
                # Extract diversity dimensions from command
                visual_intent = self._extract_visual_intent_from_command(command)
                if visual_intent:
                    visual_intents.add(visual_intent)

                element_target = self._extract_element_target_from_command(command)
                if element_target:
                    element_targets.add(element_target)

                perturbation_types.add(p.get("perturbation_type", "unknown"))
                api_calls.add(self._extract_api_call_from_command(command))

        # Calculate diversity score
        total_dimensions = 4  # visual_intent, element_target, perturbation_type, api_call
        used_dimensions = (
            len(visual_intents) + len(element_targets) + len(perturbation_types) + len(api_calls)
        )
        diversity_score = min(1.0, used_dimensions / (total_dimensions * 2))  # Normalize

        # Identify missing dimensions
        missing_dimensions = []
        if not visual_intents:
            missing_dimensions.append("visual_intent")
        if not element_targets:
            missing_dimensions.append("element_target")
        if len(perturbation_types) < 2:
            missing_dimensions.append("perturbation_type_variety")
        if len(api_calls) < 2:
            missing_dimensions.append("api_call_variety")

        # Identify overused dimensions
        overused_dimensions = []
        if len(visual_intents) == 1 and len(recent_perturbations) >= 3:
            overused_dimensions.append(f"visual_intent:{list(visual_intents)[0]}")
        if len(element_targets) == 1 and len(recent_perturbations) >= 3:
            overused_dimensions.append(f"element_target:{list(element_targets)[0]}")

        # Generate recommendations
        recommendations = []
        if diversity_score < 0.5:
            recommendations.append("Low diversity detected - try different visual modification types")
        if "visual_intent" in missing_dimensions:
            recommendations.append("Consider adding visual modifications (theme, color, typography, layout)")
        if "element_target" in missing_dimensions:
            recommendations.append("Consider targeting different UI elements")
        if "perturbation_type_variety" in missing_dimensions:
            recommendations.append("Try different perturbation types beyond current ones")

        return {
            "diversity_score": diversity_score,
            "missing_dimensions": missing_dimensions,
            "overused_dimensions": overused_dimensions,
            "recommendations": recommendations,
            "used_dimensions": {
                "visual_intents": list(visual_intents),
                "element_targets": list(element_targets),
                "perturbation_types": list(perturbation_types),
                "api_calls": list(api_calls),
            },
        }

    def _extract_visual_intent_from_command(self, command: str) -> str:
        """Extract visual intent from command (similar to trajectory_generator logic)"""
        command_lower = command.lower()

        if any(theme_word in command_lower for theme_word in ["theme", "gtk-theme", "qt-theme"]):
            return "theme"
        elif any(
            color_word in command_lower for color_word in ["color", "background", "border", "rgba", "#"]
        ):
            return "color"
        elif any(font_word in command_lower for font_word in ["font", "text", "typography", "size"]):
            return "typography"
        elif any(
            layout_word in command_lower
            for layout_word in ["margin", "padding", "spacing", "position", "size"]
        ):
            return "layout"
        elif any(css_word in command_lower for css_word in ["css", "style", "inject", "modify"]):
            return "styling"
        elif any(sys_word in command_lower for sys_word in ["wallpaper", "desktop", "system", "gsettings"]):
            return "system"

        return ""

    def _extract_element_target_from_command(self, command: str) -> str:
        """Extract element target from command (similar to trajectory_generator logic)"""
        command_lower = command.lower()

        element_targets = [
            "button",
            "input",
            "text",
            "link",
            "menu",
            "toolbar",
            "sidebar",
            "header",
            "footer",
            "navigation",
            "tab",
            "dialog",
            "modal",
            "form",
            "table",
            "list",
            "grid",
            "panel",
            "container",
        ]

        for element in element_targets:
            if element in command_lower:
                return element

        if "body" in command_lower:
            return "body"
        elif "html" in command_lower:
            return "html"
        elif "document" in command_lower:
            return "document"

        return ""

    def _extract_api_call_from_command(self, command: str) -> str:
        """Extract API call type from command"""
        import re

        api_match = re.search(r"execute_(\w+)_command", command.lower())
        return api_match.group(1) if api_match else "unknown"

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
    """Focused catalog of UI element-level visual and semantic content variations"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.catalog = self._build_focused_catalog()

    def _build_focused_catalog(self) -> Dict[str, Any]:
        """Build focused catalog emphasizing UI element-level variations"""
        return {
            "ui_element_variations": self._load_ui_element_variations(),
            "semantic_content_variations": self._load_semantic_content_variations(),
            "visual_theme_variations": self._load_visual_theme_variations(),
            "app_specific_operations": self._load_app_specific_operations(),
            "system_integration": self._load_system_integration(),
            "perturbation_categories": [
                "ui_element_visual",
                "ui_element_semantic",
                "ui_element_layout",
                "ui_element_interaction",
                "content_text_variation",
                "content_data_variation",
                "theme_color_variation",
                "theme_font_variation",
                "theme_layout_variation",
                "accessibility_variation",
                "system_theme_variation",
            ],
        }

    def _load_ui_element_variations(self) -> Dict[str, List[str]]:
        """Load UI element-level visual and semantic variations"""
        return {
            "button_variations": [
                # Button color variations using CSS injection
                "execute_css_injection('button { background-color: #ff6b6b !important; border-radius: 12px !important; box-shadow: 0 4px 8px rgba(0,0,0,0.3) !important; }', {'target_app': 'chrome'})",
                "execute_css_injection('button { background-color: #4ecdc4 !important; border-radius: 8px !important; border: 2px solid #45b7d1 !important; }', {'target_app': 'chrome'})",
                "execute_css_injection('button { background-color: #45b7d1 !important; border-radius: 20px !important; transform: scale(1.05) !important; }', {'target_app': 'chrome'})",
                # Button text variations using DOM modification
                "execute_dom_modification('document.querySelectorAll(\"button\").forEach(btn => btn.textContent = btn.textContent.toUpperCase())', {'target_app': 'chrome'})",
                "execute_dom_modification('document.querySelectorAll(\"button\").forEach(btn => btn.textContent = btn.textContent.toLowerCase())', {'target_app': 'chrome'})",
                # Button border variations
                "execute_css_injection('button { border-radius: 20px !important; border: 3px solid #e74c3c !important; }', {'target_app': 'chrome'})",
                "execute_css_injection('button { border-radius: 5px !important; border: 1px solid #95a5a6 !important; }', {'target_app': 'chrome'})",
            ],
            "input_variations": [
                # Input placeholder variations
                'execute_dom_modification(\'document.querySelectorAll("input[type=\\"text\\"], input[type=\\"email\\"], textarea").forEach(input => input.placeholder = "Enter your text here...")\', {\'target_app\': \'chrome\'})',
                'execute_dom_modification(\'document.querySelectorAll("input[type=\\"text\\"], input[type=\\"email\\"], textarea").forEach(input => input.placeholder = "Type something...")\', {\'target_app\': \'chrome\'})',
                # Input border variations
                "execute_css_injection('input[type=\"text\"], input[type=\"email\"], textarea { border: 2px solid #4ecdc4 !important; border-radius: 8px !important; padding: 12px !important; }', {'target_app': 'chrome'})",
                "execute_css_injection('input[type=\"text\"], input[type=\"email\"], textarea { border: 3px solid #e74c3c !important; border-radius: 15px !important; background-color: #f8f9fa !important; }', {'target_app': 'chrome'})",
            ],
            "link_variations": [
                # Link color and style variations
                "execute_css_injection('a { color: #e74c3c !important; text-decoration: underline !important; font-weight: bold !important; }', {'target_app': 'chrome'})",
                "execute_css_injection('a { color: #3498db !important; text-decoration: none !important; border-bottom: 2px solid #3498db !important; }', {'target_app': 'chrome'})",
                "execute_css_injection('a { color: #2ecc71 !important; text-decoration: line-through !important; font-style: italic !important; }', {'target_app': 'chrome'})",
            ],
            "image_variations": [
                # Image filter and styling variations
                "execute_dom_modification('document.querySelectorAll(\"img\").forEach(img => { img.style.filter = \"hue-rotate(180deg)\"; img.style.borderRadius = \"10px\"; })', {'target_app': 'chrome'})",
                "execute_dom_modification('document.querySelectorAll(\"img\").forEach(img => { img.style.filter = \"sepia(100%)\"; img.style.borderRadius = \"50%\"; })', {'target_app': 'chrome'})",
                "execute_dom_modification('document.querySelectorAll(\"img\").forEach(img => { img.style.filter = \"blur(2px)\"; img.style.opacity = \"0.8\"; })', {'target_app': 'chrome'})",
            ],
            "text_variations": [
                # Text styling variations
                "execute_css_injection('p, div, span { font-family: \"Times New Roman\", serif !important; font-size: 18px !important; line-height: 1.6 !important; }', {'target_app': 'chrome'})",
                "execute_css_injection('h1, h2, h3, h4, h5, h6 { color: #8e44ad !important; text-shadow: 2px 2px 4px rgba(0,0,0,0.3) !important; }', {'target_app': 'chrome'})",
                "execute_css_injection('p, div, span { font-family: \"Courier New\", monospace !important; font-size: 14px !important; letter-spacing: 1px !important; }', {'target_app': 'chrome'})",
            ],
        }

    def _load_semantic_content_variations(self) -> Dict[str, List[str]]:
        """Load semantic content variations for text and data"""
        return {
            "text_semantic_variations": [
                # Text content variations using LibreOffice Writer
                "execute_uno_command('WriterTools.modify_text_content(\"the\", \"THE\")', {'target_app': 'libreoffice_writer'})",
                "execute_uno_command('WriterTools.modify_text_content(\"and\", \"AND\")', {'target_app': 'libreoffice_writer'})",
                "execute_uno_command('WriterTools.modify_text_content(\"is\", \"IS\")', {'target_app': 'libreoffice_writer'})",
                # Text formatting variations
                "execute_uno_command('WriterTools.set_font_size(14)', {'target_app': 'libreoffice_writer'})",
                "execute_uno_command('WriterTools.set_font_size(18)', {'target_app': 'libreoffice_writer'})",
                "execute_uno_command('WriterTools.set_font_family(\"Arial\")', {'target_app': 'libreoffice_writer'})",
            ],
            "data_semantic_variations": [
                # Data content variations using LibreOffice Calc
                "execute_uno_command('CalcTools.modify_cell_content(\"A1\", \"Modified Data\")', {'target_app': 'libreoffice_calc'})",
                "execute_uno_command('CalcTools.modify_cell_content(\"B1\", \"Updated Value\")', {'target_app': 'libreoffice_calc'})",
                "execute_uno_command('CalcTools.format_range(\"A1:C10\", \"background_color\", \"#f8f9fa\")', {'target_app': 'libreoffice_calc'})",
                "execute_uno_command('CalcTools.format_range(\"A1:C10\", \"text_color\", \"#495057\")', {'target_app': 'libreoffice_calc'})",
            ],
            "web_content_variations": [
                # Web content variations using DOM modification
                "execute_dom_modification('document.querySelectorAll(\"h1, h2, h3\").forEach(heading => heading.textContent = heading.textContent.toUpperCase())', {'target_app': 'chrome'})",
                "execute_dom_modification('document.querySelectorAll(\"p\").forEach(p => p.textContent = p.textContent.replace(/the/gi, \"THE\"))', {'target_app': 'chrome'})",
                "execute_dom_modification('document.querySelectorAll(\"span\").forEach(span => span.textContent = span.textContent.replace(/and/gi, \"AND\"))', {'target_app': 'chrome'})",
            ],
        }

    def _load_visual_theme_variations(self) -> Dict[str, List[str]]:
        """Load visual theme variations for system-wide and app-specific themes"""
        return {
            "system_theme_variations": [
                # System theme changes
                "execute_bash_command('gsettings set org.gnome.desktop.interface gtk-theme \"Adwaita-dark\"')",
                "execute_bash_command('gsettings set org.gnome.desktop.interface gtk-theme \"Adwaita\"')",
                "execute_bash_command('gsettings set org.gnome.desktop.interface gtk-theme \"HighContrast\"')",
                # System font changes
                "execute_bash_command('gsettings set org.gnome.desktop.interface font-name \"Liberation Sans 14\"')",
                "execute_bash_command('gsettings set org.gnome.desktop.interface font-name \"Liberation Serif 16\"')",
                "execute_bash_command('gsettings set org.gnome.desktop.interface font-name \"Liberation Mono 12\"')",
                # System icon theme changes
                "execute_bash_command('gsettings set org.gnome.desktop.interface icon-theme \"Papirus-Dark\"')",
                "execute_bash_command('gsettings set org.gnome.desktop.interface icon-theme \"Papirus\"')",
                "execute_bash_command('gsettings set org.gnome.desktop.interface icon-theme \"Adwaita\"')",
            ],
            "chrome_theme_variations": [
                # Chrome theme randomization
                "execute_theme_randomization({'target_app': 'chrome'})",
                "execute_layout_perturbation({'target_app': 'chrome'})",
                "execute_typography_randomization({'target_app': 'chrome'})",
                "execute_accessibility_perturbation({'target_app': 'chrome'})",
                # Chrome CSS theme variations
                "execute_css_injection('body { background-color: #1a1a1a !important; color: #ffffff !important; }', {'target_app': 'chrome'})",
                "execute_css_injection('body { background-color: #2d2d2d !important; color: #00ff00 !important; }', {'target_app': 'chrome'})",
                "execute_css_injection('body { background-color: #000000 !important; color: #ffff00 !important; }', {'target_app': 'chrome'})",
            ],
            "libreoffice_theme_variations": [
                # LibreOffice theme changes
                "execute_uno_command('CalcTools.set_theme(\"dark\")', {'target_app': 'libreoffice_calc'})",
                "execute_uno_command('WriterTools.set_theme(\"dark\")', {'target_app': 'libreoffice_writer'})",
                "execute_uno_command('ImpressTools.set_theme(\"dark\")', {'target_app': 'libreoffice_impress'})",
                # LibreOffice formatting variations
                "execute_uno_command('CalcTools.format_range(\"A1:C10\", \"background_color\", \"#f8f9fa\")', {'target_app': 'libreoffice_calc'})",
                "execute_uno_command('WriterTools.set_font(\"Arial\", 14)', {'target_app': 'libreoffice_writer'})",
                "execute_uno_command('ImpressTools.set_slide_background(\"#e9ecef\")', {'target_app': 'libreoffice_impress'})",
            ],
            "vlc_theme_variations": [
                # VLC theme and visual effects
                "execute_vlc_visual_effects('apply_video_filter(\"blur\")', {'target_app': 'vlc'})",
                "execute_vlc_visual_effects('apply_video_filter(\"sepia\")', {'target_app': 'vlc'})",
                "execute_vlc_visual_effects('change_aspect_ratio(\"16_9\")', {'target_app': 'vlc'})",
                "execute_vlc_visual_effects('modify_video_brightness(\"1.2\")', {'target_app': 'vlc'})",
            ],
        }

    def _load_app_specific_operations(self) -> Dict[str, List[str]]:
        """Load app-specific operations for targeted perturbations"""
        return {
            "vlc_operations": [
                # VLC visual effects
                "execute_vlc_visual_effects('apply_video_filter(\"blur\")', {'target_app': 'vlc'})",
                "execute_vlc_visual_effects('apply_video_filter(\"sepia\")', {'target_app': 'vlc'})",
                "execute_vlc_visual_effects('change_aspect_ratio(\"16_9\")', {'target_app': 'vlc'})",
                "execute_vlc_visual_effects('modify_video_brightness(\"1.2\")', {'target_app': 'vlc'})",
            ],
            "chrome_operations": [
                # Chrome visual manipulation
                "execute_chrome_visual_manipulation('inject_custom_css(\"red_theme\")', {'target_app': 'chrome'})",
                "execute_chrome_visual_manipulation('inject_custom_css(\"dark_theme\")', {'target_app': 'chrome'})",
                "execute_chrome_visual_manipulation('modify_page_colors(\"hue_rotate\")', {'target_app': 'chrome'})",
            ],
            "vscode_operations": [
                # VS Code visual manipulation
                "execute_css_injection('.monaco-editor { background-color: #1a1a1a !important; color: #ffffff !important; }', {'target_app': 'vscode'})",
                "execute_css_injection('.monaco-editor .view-line { font-family: \"Courier New\", monospace !important; font-size: 16px !important; }', {'target_app': 'vscode'})",
                "execute_css_injection('.monaco-workbench { background-color: #2d2d2d !important; }', {'target_app': 'vscode'})",
                "execute_dom_modification('document.querySelectorAll(\".monaco-editor\").forEach(editor => editor.style.filter = \"hue-rotate(180deg)\")', {'target_app': 'vscode'})",
                "execute_theme_randomization({'target_app': 'vscode'})",
                "execute_layout_perturbation({'target_app': 'vscode'})",
            ],
            "libreoffice_operations": [
                # LibreOffice visual formatting
                "execute_libreoffice_visual_formatting('randomize_cell_colors(\"A1:C10\")', {'target_app': 'libreoffice_calc'})",
                "execute_libreoffice_visual_formatting('change_font_rendering(\"Arial\")', {'target_app': 'libreoffice_writer'})",
                "execute_libreoffice_visual_formatting('modify_border_styles(\"A1:C10\")', {'target_app': 'libreoffice_calc'})",
            ],
            "system_operations": [
                # System theme coherence
                "execute_system_theme_coherence({'target_app': 'vlc'})",
                "execute_system_theme_coherence({'target_app': 'chrome'})",
                "execute_system_theme_coherence({'target_app': 'libreoffice_calc'})",
                "execute_system_theme_coherence({'target_app': 'vscode'})",
            ],
        }

    def _load_system_integration(self) -> Dict[str, List[str]]:
        """Load system integration operations"""
        return {
            "file_operations": [
                # File system operations
                "execute_file_system_manipulation({'operation': 'create_file', 'path': '/tmp/test_file.txt', 'content': 'Test content'})",
                "execute_file_system_manipulation({'operation': 'create_directory', 'path': '/tmp/test_dir'})",
            ],
            "window_management": [
                # Window management operations
                "execute_bash_command('wmctrl -r \"Calculator\" -e 0,100,100,300,200')",
                "execute_bash_command('wmctrl -r \"Chrome\" -e 0,0,0,1920,1080')",
            ],
            "terminal_operations": [
                # Terminal operations
                'execute_bash_command(\'notify-send "System Notification" "Visual change applied"\')',
                "execute_bash_command('echo \"System perturbation executed\" > /tmp/perturbation.log')",
            ],
            "network_operations": [
                # Network perturbation
                "execute_network_perturbation({'delay': 1.0})",
                "execute_network_perturbation({'delay': 2.0})",
            ],
        }

    def _load_writer_operations(self) -> Dict[str, List[str]]:
        """Load LibreOffice Writer operations using actual UNO patterns"""
        return {
            "document": [
                'execute_uno_command(\'import uno; ctx = uno.getComponentContext(); resolver = ctx.ServiceManager.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", ctx); desktop = resolver.resolve("uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext").ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", resolver.resolve("uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext")); doc = desktop.getCurrentComponent(); doc.store()\', {\'target_app\': \'libreoffice_writer\'})',
            ],
            "text_formatting": [
                'execute_uno_command(\'import uno; ctx = uno.getComponentContext(); resolver = ctx.ServiceManager.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", ctx); desktop = resolver.resolve("uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext").ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", resolver.resolve("uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext")); doc = desktop.getCurrentComponent(); text = doc.getText(); cursor = text.createTextCursor(); cursor.CharWeight = 150; text.insertString(cursor, "Modified Text", False)\', {\'target_app\': \'libreoffice_writer\'})',
                'execute_uno_command(\'import uno; ctx = uno.getComponentContext(); resolver = ctx.ServiceManager.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", ctx); desktop = resolver.resolve("uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext").ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", resolver.resolve("uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext")); doc = desktop.getCurrentComponent(); text = doc.getText(); cursor = text.createTextCursor(); cursor.CharFontName = "Times New Roman"; cursor.CharHeight = 16; text.insertString(cursor, "Styled Text", False)\', {\'target_app\': \'libreoffice_writer\'})',
            ],
            "visual_changes": [
                'execute_uno_command(\'import uno; ctx = uno.getComponentContext(); resolver = ctx.ServiceManager.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", ctx); desktop = resolver.resolve("uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext").ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", resolver.resolve("uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext")); doc = desktop.getCurrentComponent(); text = doc.getText(); cursor = text.createTextCursor(); cursor.CharColor = 0xFF0000; text.insertString(cursor, "Red Text", False)\', {\'target_app\': \'libreoffice_writer\'})',
                'execute_uno_command(\'import uno; ctx = uno.getComponentContext(); resolver = ctx.ServiceManager.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", ctx); desktop = resolver.resolve("uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext").ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", resolver.resolve("uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext")); doc = desktop.getCurrentComponent(); text = doc.getText(); cursor = text.createTextCursor(); cursor.CharBackColor = 0xFFFF00; text.insertString(cursor, "Highlighted Text", False)\', {\'target_app\': \'libreoffice_writer\'})',
            ],
        }

    def _load_impress_operations(self) -> Dict[str, List[str]]:
        """Load LibreOffice Impress operations using actual UNO patterns"""
        return {
            "presentation": [
                'execute_uno_command(\'import uno; ctx = uno.getComponentContext(); resolver = ctx.ServiceManager.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", ctx); desktop = resolver.resolve("uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext").ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", resolver.resolve("uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext")); doc = desktop.getCurrentComponent(); doc.store()\', {\'target_app\': \'libreoffice_impress\'})',
            ],
            "slide_navigation": [
                'execute_uno_command(\'import uno; ctx = uno.getComponentContext(); resolver = ctx.ServiceManager.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", ctx); desktop = resolver.resolve("uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext").ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", resolver.resolve("uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext")); doc = desktop.getCurrentComponent(); pages = doc.getDrawPages(); controller = doc.getCurrentController(); controller.setCurrentPage(pages.getByIndex(0))\', {\'target_app\': \'libreoffice_impress\'})',
            ],
            "visual_formatting": [
                'execute_uno_command(\'import uno; ctx = uno.getComponentContext(); resolver = ctx.ServiceManager.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", ctx); desktop = resolver.resolve("uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext").ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", resolver.resolve("uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext")); doc = desktop.getCurrentComponent(); pages = doc.getDrawPages(); page = pages.getByIndex(0); shape = page.createInstance("com.sun.star.drawing.TextShape"); shape.setPosition((100, 100)); shape.setSize((400, 100)); text = shape.getText(); cursor = text.createTextCursor(); cursor.CharWeight = 150; text.insertString(cursor, "Modified Slide Text", False); page.add(shape)\', {\'target_app\': \'libreoffice_impress\'})',
                'execute_uno_command(\'import uno; ctx = uno.getComponentContext(); resolver = ctx.ServiceManager.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", ctx); desktop = resolver.resolve("uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext").ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", resolver.resolve("uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext")); doc = desktop.getCurrentComponent(); pages = doc.getDrawPages(); page = pages.getByIndex(0); shape = page.createInstance("com.sun.star.drawing.TextShape"); shape.FillColor = 0xFF0000; shape.setPosition((200, 200)); shape.setSize((300, 80)); text = shape.getText(); cursor = text.createTextCursor(); text.insertString(cursor, "Red Background Text", False); page.add(shape)\', {\'target_app\': \'libreoffice_impress\'})',
            ],
        }

    def get_operations_for_app(self, app_name: str) -> Dict[str, List[str]]:
        """Get operations for specific app"""
        app_name_lower = app_name.lower()

        # Get UI element variations (available for all apps)
        ui_variations = self.catalog["ui_element_variations"]

        # Get semantic content variations (available for all apps)
        semantic_variations = self.catalog["semantic_content_variations"]

        # Get visual theme variations (available for all apps)
        theme_variations = self.catalog["visual_theme_variations"]

        # Get app-specific operations
        app_specific = self.catalog["app_specific_operations"]
        app_ops = {}

        if app_name_lower in ["vlc"]:
            app_ops["vlc_operations"] = app_specific["vlc_operations"]
        elif app_name_lower in ["chrome", "google_chrome"]:
            app_ops["chrome_operations"] = app_specific["chrome_operations"]
        elif app_name_lower in ["code", "vscode"]:
            app_ops["vscode_operations"] = app_specific["vscode_operations"]
        elif app_name_lower in ["libreoffice_calc", "libreoffice_writer", "libreoffice_impress"]:
            app_ops["libreoffice_operations"] = app_specific["libreoffice_operations"]

        # Combine all variations
        combined_ops = {
            "ui_element_variations": ui_variations,
            "semantic_content_variations": semantic_variations,
            "visual_theme_variations": theme_variations,
            "app_specific_operations": app_ops,
        }

        return combined_ops

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
                app_formatted = f"AVAILABLE OPERATIONS FOR {app_name.upper()}:\n"
                for category, operations in app_ops.items():
                    if isinstance(operations, dict):
                        # Handle nested structure (e.g., ui_element_variations)
                        for subcategory, suboperations in operations.items():
                            if isinstance(suboperations, list) and suboperations:
                                app_formatted += (
                                    f"  {category}.{subcategory}: {len(suboperations)} operations\n"
                                )
                    elif isinstance(operations, list) and operations:
                        app_formatted += f"  {category}: {len(operations)} operations\n"
                formatted_operations.append(app_formatted)
            else:
                # Include system operations if no app-specific operations found
                formatted_operations.append(
                    f"OPERATIONS FOR {app_name.upper()}:\n  system: Using system-level operations\n"
                )

        # Always include system integration operations as fallback
        system_ops = self.catalog["system_integration"]
        system_formatted = "SYSTEM INTEGRATION OPERATIONS (guaranteed to work):\n"
        for category, operations in system_ops.items():
            if isinstance(operations, list) and operations:
                system_formatted += f"  {category}: {len(operations)} operations\n"
        formatted_operations.append(system_formatted)

        # Include UI element variations
        ui_ops = self.catalog["ui_element_variations"]
        ui_formatted = "UI ELEMENT VARIATIONS:\n"
        for category, operations in ui_ops.items():
            if isinstance(operations, list) and operations:
                ui_formatted += f"  {category}: {len(operations)} operations\n"
        formatted_operations.append(ui_formatted)

        # Include semantic content variations
        semantic_ops = self.catalog["semantic_content_variations"]
        semantic_formatted = "SEMANTIC CONTENT VARIATIONS:\n"
        for category, operations in semantic_ops.items():
            if isinstance(operations, list) and operations:
                semantic_formatted += f"  {category}: {len(operations)} operations\n"
        formatted_operations.append(semantic_formatted)

        # Include visual theme variations
        theme_ops = self.catalog["visual_theme_variations"]
        theme_formatted = "VISUAL THEME VARIATIONS:\n"
        for category, operations in theme_ops.items():
            if isinstance(operations, list) and operations:
                theme_formatted += f"  {category}: {len(operations)} operations\n"
        formatted_operations.append(theme_formatted)

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

    def __init__(self, model_name: str = "gemini-2.5-flash", model_provider: str = "gemini"):
        self.model_provider = model_provider
        self.model_name = model_name

        # TODO: Change provider and model
        # self.model_provider = "openai"
        # self.model_name = "gpt-4.1-nano"

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

    def call_llm(self, prompt: str, response_schema: BaseModel = None) -> str:
        """Call LLM with prompt using the correct API format for each provider"""
        if not self.client:
            return '{"error": "Mock response - API not available"}'

        retries = 0
        max_retries = 3
        while retries < max_retries:
            try:
                retries += 1

                if self.model_provider == "gemini":
                    if response_schema:
                        config = {
                            "response_mime_type": "application/json",
                            "response_schema": response_schema,
                        }
                    else:
                        config = {}

                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=config,
                    )
                    if response and hasattr(response, "text") and response.text:
                        if response_schema and hasattr(response, "parsed"):
                            # Use the parsed Pydantic objects directly
                            return response.parsed
                        else:
                            return response.text
                    else:
                        raise ValueError("No response text from Gemini API")

                elif self.model_provider == "openai":
                    response = self.client.responses.create(
                        model=self.model_name,
                        input=prompt,
                    )
                    return response.output_text

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
        # Add general application context for all apps
        app_summary += "  [APPLICATION INTERFACE - PRIORITIZE PRIMARY HIERARCHY ELEMENTS]\n"

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
        """Generate contextual hints for UI elements - simplified and general approach"""
        hints = []

        # Basic element information
        if element.name:
            hints.append(f"[{element.name}]")

        # Element type hints
        if element.element_type:
            hints.append(f"[{element.element_type}]")

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

        # Interactive state hints
        if not element.is_enabled:
            hints.append("[disabled]")

        # Z-order blocking hints
        if window_state and all_window_states:
            if self._is_element_blocked_by_z_order(element, window_state, all_window_states):
                hints.append("[blocked by higher window]")

        # Position hints for very small elements
        if element.position:
            width = element.position.get("width", 0)
            height = element.position.get("height", 0)
            if width <= 16 and height <= 16 and not element.name:
                hints.append("[very small - likely decorative]")

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
                    response = self.call_llm(prompt, response_schema=list[ScenarioSpecForLLM])
                    if isinstance(response, list):
                        # Convert Pydantic objects to dictionaries for compatibility with existing validation logic
                        batch_scenarios = [scenario.model_dump() for scenario in response]
                    else:
                        self.logger.error("Failed to get structured response for scenarios")
                        batch_scenarios = self._generate_fallback_scenarios(
                            remaining_count, task_context, batch_idx
                        )

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
        """Create curriculum prompt with balanced guidance for invariant feature learning"""
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

        # Generate balanced operation guidance (not specific examples)
        target_apps = task_context.get("app_types", ["system"])
        operation_guidance = self._generate_balanced_operation_guidance(target_apps)

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

OPERATION GUIDANCE FOR INVARIANT FEATURE LEARNING:
{operation_guidance}

APP-SPECIFIC GUIDANCE FOR INVARIANT FEATURE LEARNING:
{app_specific_guidance}

AVAILABLE OPERATIONS:
{task_context["available_operations"]}

INVARIANT FEATURE LEARNING FOCUS:
1. PRIORITIZE UI ELEMENT-LEVEL PERTURBATIONS:
   - Target specific UI elements (buttons, inputs, links, images, text, menus)
   - Focus on visual properties that don't affect functionality
   - Create scenarios where the same action works despite visual changes
   - Emphasize element recognition across different visual styles

2. VISUAL INVARIANCE LEARNING OBJECTIVES:
   - Color invariance: Same element with different colors
   - Shape invariance: Same element with different borders/radius
   - Size invariance: Same element with different dimensions
   - Typography invariance: Same element with different fonts
   - Layout invariance: Same element in different positions
   - Theme invariance: Same element with different themes

3. CONCRETE LEARNING SCENARIOS:
   - Button recognition: Submit button with different colors/shapes
   - Input field recognition: Text inputs with different borders/styles
   - Navigation recognition: Links with different visual treatments
   - Content recognition: Text with different fonts/sizes
   - Icon recognition: Images with different filters/effects

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
3. Use ONLY validated API calls: execute_bash_command, execute_python_command, execute_css_injection, execute_dom_modification, execute_theme_randomization, execute_layout_perturbation, execute_typography_randomization, execute_animation_effects, execute_accessibility_perturbation, execute_uno_command, execute_js_on_page, execute_python_execution, execute_javascript_injection, execute_bash_automation, execute_playwright_automation, execute_file_system_manipulation, execute_network_perturbation, execute_system_integration, execute_vlc_visual_effects, execute_chrome_visual_manipulation, execute_libreoffice_visual_formatting, execute_system_theme_coherence
4. Commands MUST be syntactically correct and tested for Ubuntu environment
5. Focus on perturbations that maintain target element accessibility
6. Ensure commands are feasible for the target application
7. Prefer simple, reliable commands over complex ones
8. Include specific parameters and values that work in Ubuntu environment
9. CREATE ORIGINAL SCENARIOS - Do not copy the guidance examples exactly

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
- Focus on meaningful visual changes for learning objectives
- Create original scenarios that demonstrate understanding of invariant learning principles
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

    def _generate_balanced_operation_guidance(self, target_apps: List[str]) -> str:
        """Generate balanced operation guidance without specific examples to prevent copying"""
        guidance_parts = []

        for app in target_apps:
            app_lower = app.lower()

            if app_lower in ["chrome", "google_chrome"]:
                guidance_parts.append("""
CHROME OPERATIONS GUIDANCE:
- Use execute_css_injection for UI element styling (buttons, inputs, links, text)
- Use execute_dom_modification for content changes and element manipulation
- Use execute_theme_randomization for overall theme changes
- Use execute_layout_perturbation for spacing and positioning changes
- Use execute_typography_randomization for font and text styling
- Focus on visual properties: colors, borders, shadows, fonts, spacing
- Target specific selectors: button, input, a, img, p, div, span
- Maintain functionality while changing appearance
""")

            elif app_lower in ["libreoffice_calc", "libreoffice_writer", "libreoffice_impress"]:
                guidance_parts.append("""
LIBREOFFICE OPERATIONS GUIDANCE:
- Use execute_uno_command for LibreOffice-specific operations
- Focus on visual formatting: colors, fonts, borders, backgrounds
- Target text elements, shapes, and formatting properties
- Use UNO API patterns for LibreOffice manipulation
- Maintain document functionality while changing appearance
- Consider slide/page navigation and content modification
""")

            elif app_lower in ["vlc"]:
                guidance_parts.append("""
VLC OPERATIONS GUIDANCE:
- Use execute_vlc_visual_effects for video filter changes
- Use execute_system_theme_coherence for system-level changes
- Focus on visual effects: blur, sepia, brightness, aspect ratio
- Consider system theme changes that affect VLC appearance
- Maintain video playback functionality
""")

            elif app_lower in ["system"]:
                guidance_parts.append("""
SYSTEM OPERATIONS GUIDANCE:
- Use execute_bash_command for system-level changes
- Use gsettings for theme, font, and desktop changes
- Use notify-send for system notifications
- Focus on desktop themes, fonts, wallpapers, and system settings
- Maintain system stability and accessibility
""")

        return "\n".join(guidance_parts)

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

            response = self.call_llm(prompt, response_schema=TaskAnalysis)

            if isinstance(response, TaskAnalysis):
                # Convert Pydantic model to dict for validation
                result = response.model_dump()
                validated_result = self._validate_llm_task_analysis(result)
                if validated_result:
                    return validated_result
                else:
                    self.logger.exception("LLM task analysis result validation failed")
            else:
                self.logger.error("Failed to get structured response from LLM")

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

    def _format_operations_for_prompt(self, app_operations: Dict[str, Any]) -> str:
        """Format operations for LLM prompt efficiently"""
        if not app_operations:
            return "No specific operations available"

        formatted_parts = []
        for category, operations in app_operations.items():
            if isinstance(operations, dict):
                # Handle nested structure (e.g., ui_element_variations)
                formatted_parts.append(f"{category.upper()}:")
                for subcategory, suboperations in operations.items():
                    if isinstance(suboperations, list) and suboperations:
                        formatted_parts.append(f"  {subcategory}:")
                        for operation in suboperations[:3]:  # Limit to 3 operations per subcategory
                            formatted_parts.append(f"    - {operation}")
            elif isinstance(operations, list) and operations:
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

    def _format_diversity_analysis_for_prompt(self, diversity_analysis: Dict[str, Any]) -> str:
        """Format diversity analysis for LLM prompt"""
        if not diversity_analysis:
            return "DIVERSITY ANALYSIS: No analysis available"

        parts = []

        # Diversity score
        score = diversity_analysis.get("diversity_score", 0.0)
        parts.append(f"DIVERSITY SCORE: {score:.2f}/1.0")

        # Missing dimensions
        missing = diversity_analysis.get("missing_dimensions", [])
        if missing:
            parts.append(f"MISSING DIMENSIONS: {', '.join(missing)}")

        # Overused dimensions
        overused = diversity_analysis.get("overused_dimensions", [])
        if overused:
            parts.append(f"OVERUSED DIMENSIONS: {', '.join(overused)}")

        # Recommendations
        recommendations = diversity_analysis.get("recommendations", [])
        if recommendations:
            parts.append("DIVERSITY RECOMMENDATIONS:")
            for rec in recommendations:
                parts.append(f"  - {rec}")

        # Used dimensions summary
        used_dims = diversity_analysis.get("used_dimensions", {})
        if used_dims:
            parts.append("CURRENT DIVERSITY:")
            for dim_type, values in used_dims.items():
                if values:
                    parts.append(f"  - {dim_type}: {', '.join(values)}")

        return "\n".join(parts) if parts else "DIVERSITY ANALYSIS: No recent perturbations"

    def _get_llm_decision_with_context(
        self, execution_context: ExecutionContext, scenario_spec: ScenarioSpec
    ) -> Dict[str, Any]:
        """Get LLM decision with procedural memory context and retries"""
        try:
            # Determine task progress based on step index
            task_progress = self._determine_task_progress(
                execution_context.step_idx, execution_context.task_instruction, execution_context.total_steps
            )

            target_app = scenario_spec.target_app

            # Get rich procedural memory context
            procedural_context = self.procedural_memory.get_context_for_decision(
                target_app, execution_context.step_idx, task_progress
            )

            # Format procedural memory context for prompt
            memory_context = self._format_procedural_memory_context(procedural_context)

            # Get app-specific strategy
            app_strategy = self._get_app_specific_strategy(target_app, execution_context)

            # Get available operations for target app (optimized)
            app_operations = self.operation_catalog.get_operations_for_app(target_app.lower())
            formatted_operations = self._format_operations_for_prompt(app_operations)

            # Build app-specific context
            app_context = self._build_app_specific_context(app_strategy, target_app)

            # Build app-specific prompt
            prompt = self._build_app_specific_prompt(
                target_app,
                execution_context,
                scenario_spec,
                app_context,
                formatted_operations,
                memory_context,
                procedural_context,
            )

            response = self.call_llm(prompt, response_schema=PerturbationDecision)

            if isinstance(response, PerturbationDecision):
                result = response.model_dump()
            else:
                self.logger.error("Failed to get structured response from LLM")
                return self._generate_fallback_response(target_app)

            is_valid, errors = self.verifier.verify_perturbation_decision(result)
            if not is_valid:
                self.logger.error(f"Validation failed: {', '.join(errors)}")
                return self._generate_fallback_response(target_app)

            return result

        except Exception as e:
            self.logger.error(f"Error in _get_llm_decision_with_context: {e}")
            return self._create_error_fallback_response(target_app, str(e))

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

    def _build_app_specific_prompt(
        self,
        target_app: str,
        execution_context: ExecutionContext,
        scenario_spec: ScenarioSpec,
        app_context: str,
        formatted_operations: str,
        memory_context: str,
        procedural_context: Dict[str, Any],
    ) -> str:
        """Build app-specific prompt with proper f-string handling"""

        # Get app-specific examples
        app_examples = self._get_app_specific_examples(target_app)

        prompt = f"""
{PROMPT_CONSTANTS["perturbation_role"]}

CURRENT EXECUTION CONTEXT:
Step: {execution_context.step_idx}
Next Action: {execution_context.current_action}
Task: {execution_context.task_instruction}
App States: {self._format_app_states_for_decision(execution_context.window_states)}

SCENARIO SPECIFICATION:
Target App: {target_app}
Trigger: {scenario_spec.perturbation_trigger}
Available Actions: {scenario_spec.available_perturbation_actions}
Learning Objectives: {scenario_spec.learning_objectives}
Perturbation Types: {[pt.value for pt in scenario_spec.perturbation_types]}

APP-SPECIFIC STRATEGY FOR {target_app.upper()}:
{app_context}

AVAILABLE OPERATIONS FOR {target_app.upper()}:
{formatted_operations}

PROCEDURAL MEMORY CONTEXT:
{memory_context}

DIVERSITY ANALYSIS:
{self._format_diversity_analysis_for_prompt(procedural_context.get("diversity_analysis", {}))}

COHERENCE REQUIREMENTS:
1. Build on previous successful perturbations in the trajectory
2. Create meaningful visual impact for learning objectives
3. Maintain application functionality and accessibility
4. Use concrete, executable commands with specific parameters
5. Consider app-specific visual perturbation opportunities

DIVERSITY REQUIREMENTS:
1. AVOID repeating the same visual modification type (theme, color, typography, layout, styling, system)
2. AVOID targeting the same UI elements repeatedly
3. PRIORITIZE missing diversity dimensions identified in the analysis
4. If diversity score is low, try different perturbation approaches
5. Balance between coherence and diversity - don't sacrifice learning for novelty

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
1. Use ONLY validated API calls: execute_bash_command, execute_python_command, execute_css_injection, execute_dom_modification, execute_theme_randomization, execute_layout_perturbation, execute_typography_randomization, execute_animation_effects, execute_accessibility_perturbation, execute_uno_command, execute_js_on_page, execute_python_execution, execute_javascript_injection, execute_bash_automation, execute_playwright_automation, execute_file_system_manipulation, execute_network_perturbation, execute_system_integration, execute_vlc_visual_effects, execute_chrome_visual_manipulation, execute_libreoffice_visual_formatting, execute_system_theme_coherence
2. Commands MUST be syntactically correct and executable
3. Include specific parameters and values that work in Ubuntu environment
4. Test commands mentally before suggesting them
5. Prefer simple, reliable commands over complex ones
6. Ensure commands maintain target element accessibility
7. Focus on UI element-level visual variations for invariant feature learning

SAFE COMMAND EXAMPLES:
{app_examples}

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
    "api_call": "execute_bash_command|execute_python_command|execute_css_injection|execute_dom_modification|execute_theme_randomization|execute_layout_perturbation|execute_typography_randomization|execute_animation_effects|execute_accessibility_perturbation|execute_uno_command|execute_js_on_page|execute_python_execution|execute_javascript_injection|execute_bash_automation|execute_playwright_automation|execute_file_system_manipulation|execute_network_perturbation|execute_system_integration|execute_vlc_visual_effects|execute_chrome_visual_manipulation|execute_libreoffice_visual_formatting|execute_system_theme_coherence",
    "generated_command": "complete_executable_command_with_specific_parameters",
    "parameters": {{
        "target_app": "{target_app}",
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
        return prompt

    def _get_app_specific_examples(self, target_app: str) -> str:
        """Get app-specific command examples with proper f-string handling"""
        app_lower = target_app.lower()

        if app_lower in ["chrome", "google-chrome", "chromium"]:
            return """# Chrome Visual Variations (PRIORITY for invariant feature learning)
- Chrome: execute_css_injection('button {{ background-color: #ff6b6b !important; border-radius: 12px !important; box-shadow: 0 4px 8px rgba(0,0,0,0.3) !important; }}', {{"target_app": "chrome"}})
- Chrome: execute_css_injection('input[type="text"], input[type="email"], textarea {{ border: 2px solid #4ecdc4 !important; border-radius: 8px !important; padding: 12px !important; }}', {{"target_app": "chrome"}})
- Chrome: execute_css_injection('a {{ color: #e74c3c !important; text-decoration: underline !important; font-weight: bold !important; }}', {{"target_app": "chrome"}})
- Chrome: execute_dom_modification('document.querySelectorAll("button").forEach(btn => {{ btn.style.backgroundColor = "#3498db"; btn.style.transform = "scale(1.05)"; }})', {{"target_app": "chrome"}})
- Chrome: execute_dom_modification('document.querySelectorAll("img").forEach(img => {{ img.style.filter = "hue-rotate(180deg)"; img.style.borderRadius = "10px"; }})', {{"target_app": "chrome"}})
- Chrome: execute_theme_randomization({{"target_app": "chrome"}})
- Chrome: execute_layout_perturbation({{"target_app": "chrome"}})
- Chrome: execute_typography_randomization({{"target_app": "chrome"}})"""

        elif app_lower in ["libreoffice_calc", "libreoffice_writer", "libreoffice_impress"]:
            return """# LibreOffice Visual Variations
- LibreOffice: execute_uno_command('CalcTools.set_theme("dark")', {{"target_app": "libreoffice_calc"}})
- LibreOffice: execute_uno_command('CalcTools.format_range("A1:C10", "background_color", "#f8f9fa")', {{"target_app": "libreoffice_calc"}})
- LibreOffice: execute_uno_command('WriterTools.set_font("Arial", 14)', {{"target_app": "libreoffice_writer"}})
- LibreOffice: execute_libreoffice_visual_formatting('change_toolbar_colors', {{"target_app": "libreoffice_calc"}})"""

        elif app_lower == "vlc":
            return """# VLC Visual Variations
- VLC: execute_vlc_visual_effects('apply_video_filter("blur")', {{"target_app": "vlc"}})
- VLC: execute_vlc_visual_effects('change_aspect_ratio("16_9")', {{"target_app": "vlc"}})
- VLC: execute_vlc_visual_effects('modify_interface_theme', {{"target_app": "vlc"}})"""

        elif app_lower in ["code", "vscode"]:
            return """# VS Code Visual Variations
- VS Code: execute_vscode_theme_change('dark_plus', {{"target_app": "vscode"}})
- VS Code: execute_vscode_color_customization('editor.background', '#1e1e1e', {{"target_app": "vscode"}})"""

        else:
            return """# System-Level Variations
- System: execute_bash_command('gsettings set org.gnome.desktop.interface gtk-theme "Adwaita-dark"')
- System: execute_bash_command('gsettings set org.gnome.desktop.interface font-name "Liberation Sans 14"')
- System: execute_bash_command('notify-send "Visual Change" "UI element styling applied"')
- System: execute_system_theme_coherence({{"target_app": "{target_app}"}})"""


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

        GENERAL ELEMENT IDENTIFICATION RULES:
        1. CONTEXT RELEVANCE: Prioritize elements that match the action's context and intent
        2. ELEMENT TYPE APPROPRIATENESS: Match element types to action types (e.g., "CLICK" actions should prioritize clickable elements)
        3. VISIBILITY AND INTERACTIVITY: Prioritize visible, enabled, and interactive elements
        4. TEXT MATCHING: Elements with text content matching the action description should be prioritized
        5. POSITION REASONABLENESS: Elements with coordinates (0,0) or outside screen bounds should be deprioritized
        6. HIERARCHY AWARENESS: Consider element hierarchy and parent context when making decisions

        ELEMENT PRIORITIZATION GUIDELINES:
        - Elements with clear, descriptive names should be prioritized over generic elements
        - Elements in primary interface areas (menus, toolbars, navigation) should be prioritized
        - Elements with specific roles (button, link, input) should be prioritized over generic elements
        - Elements with meaningful text content should be prioritized over empty elements
        - Elements that are currently visible and interactive should be prioritized

        DISAMBIGUATION STRATEGIES:
        - If multiple elements have the same name, consider ALL of them as potential candidates
        - Elements marked as "[collapsed - likely invisible]" or "[hidden - likely invisible]" should be deprioritized
        - Elements marked as "[blocked by higher window]" should be deprioritized
        - Elements marked as "[disabled]" should be deprioritized
        - Very small elements (≤16x16 pixels) without names should be deprioritized unless they have interactive properties

        RANKING APPROACH:
        1. Direct text matches with the action description
        2. Element type appropriateness for the action
        3. Element visibility and interactivity
        4. Element context and hierarchy
        5. Element position and size reasonableness

        Return JSON array with element identifiers, ranked by likelihood from 0.00 to 1.00 and only return elements with confidence values greater than 0.70:
        [
            {{
                "name": "element_name",
                "element_type": "element_type",
                "app_name": "app_name",
                "confidence": 0.95,
                "reasoning": "detailed_reasoning_for_element_selection_including_context_and_coordinate_analysis"
            }},
            {{
                "name": "alternative_element_name",
                "element_type": "element_type",
                "app_name": "app_name",
                "confidence": 0.75,
                "reasoning": "alternative_reasoning_with_context_considerations"
            }}
        ]
        """

        response = self.call_llm(prompt, response_schema=list[ElementCandidate])

        if not isinstance(response, list):
            return []

        # Convert Pydantic objects to dictionaries for compatibility
        return [candidate.model_dump() for candidate in response]


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
