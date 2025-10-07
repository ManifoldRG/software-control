"""
Clean LLM Services: Simplified interfaces for LLM interactions
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

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
from perturbation_engine.tools.autoglm_integration import AutoglmPerturbationGenerator


class CleanLLM:
    """Clean, simplified LLM interface"""

    def __init__(self, model_name: str = "gemini-2.0-flash-lite"):
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

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=0)),
            )
            return response.text
        except Exception as e:
            self.logger.error(f"Error calling LLM: {e}")
            return '{"error": "LLM call failed"}'

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


class CleanCurriculumLLM(CleanLLM):
    """Enhanced curriculum generation with realistic GUI perturbations"""

    def __init__(self, model_name: str = "gemini-2.0-flash-lite"):
        super().__init__(model_name)
        self.verifier = LLMOutputVerifier()

    def generate_scenario_specs(
        self,
        seed_trajectory: SeedTrajectory,
        app_states: List[Dict[str, Any]],
        curriculum_config: CurriculumConfig,
    ) -> List[ScenarioSpec]:
        """Generate diverse, realistic perturbation scenarios"""

        # Analyze task complexity and app types
        task_analysis = self._analyze_task_complexity(seed_trajectory.task_instruction)
        app_analysis = self._analyze_app_states(app_states)

        prompt = f"""
        You are an expert in GUI automation and computer vision. Generate {curriculum_config.scenario_count} realistic perturbation scenarios that will teach visual invariance while maintaining task feasibility.

        TASK CONTEXT:
        Task: {seed_trajectory.task_instruction}
        Task Complexity: {task_analysis["complexity"]}
        Task Domain: {task_analysis["domain"]}
        Critical Elements: {task_analysis["critical_elements"]}

        AVAILABLE APPLICATIONS:
        {self._format_app_states_for_curriculum(app_states)}

        PERTURBATION REQUIREMENTS:
        1. Each scenario must maintain the original task's feasibility
        2. Target elements must remain reachable after perturbation
        3. Use realistic GUI changes that increase visual complexity
        4. Focus on teaching visual invariance and robustness
        5. Consider task-specific constraints

        AVAILABLE PERTURBATION CATEGORIES:
        - THEME: Color schemes, dark/light modes, visual themes
        - LAYOUT: Spacing, positioning, window arrangements, UI density
        - TYPOGRAPHY: Font changes, text styling, readability variations
        - CONTENT_VARIATION: Data changes, file names, text content
        - UI_INJECTION: Adding elements, popups, notifications
        - WINDOW_MANAGEMENT: Window states, focus changes, positioning
        - FILE_OPERATIONS: File system changes, directory structures

        Return JSON array with scenario objects:
        {{
            "scenario_id": "unique_identifier",
            "target_app": "specific_app_name",
            "perturbation_trigger": "specific_condition_when_to_apply",
            "available_perturbation_actions": "detailed_autoglm_v_command_sequence",
            "learning_objectives": "specific_learning_goal_for_visual_invariance",
            "target_components": ["specific_ui_elements_to_target"],
            "perturbation_types": ["primary_perturbation_category"],
            "perturbation_intensity": "low|medium|high",
            "maintains_functionality": true,
            "realistic_scenario": "explanation_of_realistic_context"
        }}

        Generate diverse scenarios that cover different perturbation types and applications.
        """

        response = self.call_llm(prompt)
        scenarios_data = self.extract_json(response)

        # Validate and enhance scenarios
        scenario_specs = []
        for i, scenario_data in enumerate(scenarios_data):
            try:
                # Sanitize and verify scenario data
                sanitized_data = self.verifier.sanitize_scenario_data(scenario_data)
                is_valid, errors = self.verifier.verify_scenario_spec(sanitized_data)

                if not is_valid:
                    self.logger.warning(f"Scenario {i} validation failed: {errors}")
                    # Try to enhance with defaults
                    enhanced_data = self.verifier.enhance_scenario_with_defaults(sanitized_data)
                    validated_scenario = self._validate_and_enhance_scenario(
                        enhanced_data, task_analysis, app_analysis
                    )
                else:
                    validated_scenario = self._validate_and_enhance_scenario(
                        sanitized_data, task_analysis, app_analysis
                    )

                if validated_scenario:
                    scenario_specs.append(validated_scenario)
            except Exception as e:
                self.logger.error(f"Error processing scenario {i}: {e}")
                continue

        # Ensure we have enough scenarios
        while len(scenario_specs) < curriculum_config.scenario_count:
            fallback_scenario = self._create_fallback_scenario(seed_trajectory, app_analysis)
            scenario_specs.append(fallback_scenario)

        return scenario_specs[: curriculum_config.scenario_count]

    def _analyze_task_complexity(self, task_instruction: str) -> Dict[str, Any]:
        """Analyze task to understand complexity and constraints"""
        prompt = f"""
        Analyze this GUI automation task for perturbation planning:

        Task: {task_instruction}

        Return JSON with:
        {{
            "complexity": "simple|moderate|complex",
            "domain": "office|web|multimedia|development|system",
            "critical_elements": ["list_of_critical_ui_elements"],
            "constraints": ["list_of_task_constraints"],
            "perturbation_sensitivity": "low|medium|high"
        }}
        """

        response = self.call_llm(prompt)
        result = self.extract_json(response)

        if isinstance(result, list) and len(result) > 0:
            result = result[0]

        return (
            result
            if isinstance(result, dict)
            else {
                "complexity": "moderate",
                "domain": "general",
                "critical_elements": [],
                "constraints": [],
                "perturbation_sensitivity": "medium",
            }
        )

    def _analyze_app_states(self, app_states: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze app states to understand available applications and elements"""
        app_info = {}

        for app_state in app_states:
            app_name = app_state.get("app_name", "unknown")
            elements = app_state.get("elements", [])

            app_info[app_name] = {
                "element_count": len(elements),
                "element_types": list({elem.get("element_type", "unknown") for elem in elements}),
                "has_text_elements": any(elem.get("text", "") for elem in elements),
                "has_interactive_elements": any(
                    elem.get("element_type", "").lower() in ["button", "link", "input", "menu"]
                    for elem in elements
                ),
            }

        return app_info

    def _format_app_states_for_curriculum(self, app_states: List[Dict[str, Any]]) -> str:
        """Format app states for curriculum generation"""
        if not app_states:
            return "No applications detected"

        formatted = []
        for app_state in app_states:
            app_name = app_state.get("app_name", "Unknown")
            elements = app_state.get("elements", [])

            element_summary = f"  - {len(elements)} elements"
            if elements:
                element_types = list({elem.get("element_type", "unknown") for elem in elements[:5]})
                element_summary += f" (types: {', '.join(element_types)})"

            formatted.append(f"App: {app_name}\n{element_summary}")

        return "\n".join(formatted)

    def _validate_and_enhance_scenario(
        self, scenario_data: Dict[str, Any], task_analysis: Dict[str, Any], app_analysis: Dict[str, Any]
    ) -> Optional[ScenarioSpec]:
        """Validate and enhance scenario data"""
        try:
            # Parse perturbation types
            perturbation_types = []
            for pt_str in scenario_data.get("perturbation_types", []):
                mapped_type = PerturbationType.from_string(pt_str, default=PerturbationType.THEME)
                perturbation_types.append(mapped_type)

            if not perturbation_types:
                perturbation_types.append(PerturbationType.THEME)

            # Validate target app exists
            target_app = scenario_data.get("target_app", "unknown")
            if target_app not in app_analysis:
                self.logger.warning(f"Target app {target_app} not found in app states")
                return None

            # Enhance perturbation actions with autoglm_v integration
            enhanced_actions = self._enhance_perturbation_actions(
                scenario_data.get("available_perturbation_actions", ""), target_app, perturbation_types[0]
            )

            return ScenarioSpec(
                scenario_id=scenario_data.get("scenario_id", f"scenario_{hash(str(scenario_data))}"),
                target_app=target_app,
                perturbation_trigger=scenario_data.get("perturbation_trigger", "During task execution"),
                available_perturbation_actions=enhanced_actions,
                learning_objectives=scenario_data.get("learning_objectives", "Learn visual invariance"),
                target_components=scenario_data.get("target_components", []),
                perturbation_types=perturbation_types,
            )

        except Exception as e:
            self.logger.error(f"Error validating scenario: {e}")
            return None

    def _enhance_perturbation_actions(
        self, actions: str, target_app: str, perturbation_type: PerturbationType
    ) -> str:
        """Enhance perturbation actions with autoglm_v commands"""
        if not actions or actions.strip() == "":
            # Generate default actions based on app and perturbation type
            return self._generate_default_actions(target_app, perturbation_type)

        # Enhance existing actions with autoglm_v integration
        enhanced = f"""
# Enhanced perturbation actions for {target_app} - {perturbation_type.value}
{actions}

# Additional autoglm_v integration
from perturbation_engine.tools.autoglm_integration import AutoglmPerturbationGenerator
generator = AutoglmPerturbationGenerator()
additional_command = generator.generate_perturbation_command("{target_app}", "{perturbation_type.value}", {{}})
"""
        return enhanced.strip()

    def _generate_default_actions(self, target_app: str, perturbation_type: PerturbationType) -> str:
        """Generate default perturbation actions"""
        app_lower = target_app.lower()

        if perturbation_type == PerturbationType.THEME:
            if "chrome" in app_lower:
                return "BrowserTools.open_appearance_settings()"
            elif "code" in app_lower:
                return "CodeTools.install_extension('ms-vscode.theme-materialdark')"
            else:
                return "gsettings set org.gnome.desktop.interface gtk-theme 'Adwaita-dark'"

        elif perturbation_type == PerturbationType.LAYOUT:
            if "calc" in app_lower:
                return "CalcTools.adjust_column_width('A:C', autofit=True)"
            else:
                return "wmctrl -r :ACTIVE: -e 0,100,100,800,600"

        elif perturbation_type == PerturbationType.CONTENT_VARIATION:
            if "calc" in app_lower:
                return "CalcTools.set_cell_value('A1', 'Perturbed Data')"
            else:
                return "echo 'Content variation applied'"

        else:
            return f"# {perturbation_type.value} perturbation for {target_app}"

    def _create_fallback_scenario(
        self, seed_trajectory: SeedTrajectory, app_analysis: Dict[str, Any]
    ) -> ScenarioSpec:
        """Create fallback scenario when LLM generation fails"""
        available_apps = list(app_analysis.keys())
        target_app = available_apps[0] if available_apps else "system"

        return ScenarioSpec(
            scenario_id=f"fallback_scenario_{hash(seed_trajectory.task_instruction)}",
            target_app=target_app,
            perturbation_trigger="During task execution",
            available_perturbation_actions=self._generate_default_actions(target_app, PerturbationType.THEME),
            learning_objectives="Learn to adapt to visual changes",
            target_components=["general"],
            perturbation_types=[PerturbationType.THEME],
        )


class CleanPerturbationLLM(CleanLLM):
    """Enhanced perturbation decision making with procedural memory integration"""

    def __init__(self, model_name: str = "gemini-2.0-flash-lite"):
        super().__init__(model_name)
        self.autoglm_generator = AutoglmPerturbationGenerator()
        self.perturbation_history = []  # Track perturbation context
        self.procedural_memory = {}  # Store procedural memory for coherent perturbations
        self.verifier = LLMOutputVerifier()

    def decide_perturbation(
        self, execution_context: ExecutionContext, scenario_spec: ScenarioSpec
    ) -> Dict[str, Any]:
        """Decide whether to apply perturbation with procedural memory context"""

        # Update procedural memory with current context
        self._update_procedural_memory(execution_context, scenario_spec)

        # Get LLM decision with procedural memory context
        llm_decision = self._get_llm_decision_with_context(execution_context, scenario_spec)

        # Enhance with autoglm_v and procedural memory if perturbation should be applied
        if llm_decision.get("should_apply", False):
            enhanced_decision = self._enhance_with_procedural_memory(
                llm_decision, scenario_spec, execution_context
            )
            return enhanced_decision

        return llm_decision

    def _update_procedural_memory(self, execution_context: ExecutionContext, scenario_spec: ScenarioSpec):
        """Update procedural memory with current execution context"""
        memory_key = f"{scenario_spec.target_app}_{execution_context.step_idx}"

        self.procedural_memory[memory_key] = {
            "step_idx": execution_context.step_idx,
            "current_action": execution_context.current_action,
            "app_states": execution_context.app_states,
            "target_app": scenario_spec.target_app,
            "perturbation_types": [pt.value for pt in scenario_spec.perturbation_types],
            "timestamp": execution_context.timestamp,
            "previous_perturbations": self.perturbation_history[-3:] if self.perturbation_history else [],
        }

    def _get_llm_decision_with_context(
        self, execution_context: ExecutionContext, scenario_spec: ScenarioSpec
    ) -> Dict[str, Any]:
        """Get LLM decision with procedural memory context"""

        # Build procedural memory context
        memory_context = self._build_memory_context(scenario_spec.target_app)

        prompt = f"""
        You are an expert in GUI automation perturbation. Decide whether to apply perturbation at the current step, considering the procedural memory context.

        CURRENT EXECUTION CONTEXT:
        Step: {execution_context.step_idx}
        Action: {execution_context.current_action}
        Task: {execution_context.task_instruction}
        App States: {self._format_app_states_for_decision(execution_context.app_states)}

        SCENARIO SPECIFICATION:
        Target App: {scenario_spec.target_app}
        Trigger: {scenario_spec.perturbation_trigger}
        Available Actions: {scenario_spec.available_perturbation_actions}
        Learning Objectives: {scenario_spec.learning_objectives}
        Perturbation Types: {[pt.value for pt in scenario_spec.perturbation_types]}

        PROCEDURAL MEMORY CONTEXT:
        {memory_context}

        PERTURBATION DECISION CRITERIA:
        1. Does the current step match the perturbation trigger?
        2. Would perturbation enhance learning without breaking the task?
        3. Is the perturbation coherent with previous perturbations?
        4. Does the target app have the necessary elements for perturbation?
        5. Would the perturbation maintain task feasibility?

        Return JSON:
        {{
            "should_apply": true/false,
            "reasoning": "detailed_explanation_of_decision",
            "perturbation_type": "specific_perturbation_type",
            "parameters": {{
                "target_app": "{scenario_spec.target_app}",
                "intensity": "low|medium|high",
                "coherent_with_history": true/false,
                "maintains_functionality": true/false
            }},
            "confidence": 0.0-1.0,
            "alternative_actions": ["list_of_alternative_perturbation_options"]
        }}
        """

        response = self.call_llm(prompt)
        result = self.extract_json(response)

        if isinstance(result, list) and len(result) > 0:
            result = result[0]

        if not isinstance(result, dict):
            return {
                "should_apply": False,
                "reasoning": "Failed to parse LLM response",
                "perturbation_type": "theme",
                "parameters": {"target_app": scenario_spec.target_app},
                "confidence": 0.0,
                "alternative_actions": [],
            }

        # Verify the decision data
        is_valid, errors = self.verifier.verify_perturbation_decision(result)
        if not is_valid:
            self.logger.warning(f"Perturbation decision validation failed: {errors}")
            # Return safe default decision
            return {
                "should_apply": False,
                "reasoning": f"Validation failed: {', '.join(errors)}",
                "perturbation_type": "theme",
                "parameters": {"target_app": scenario_spec.target_app},
                "confidence": 0.0,
                "alternative_actions": [],
            }

        return result

    def _build_memory_context(self, target_app: str) -> str:
        """Build procedural memory context for the target app"""
        app_memories = [mem for mem in self.procedural_memory.values() if mem["target_app"] == target_app]

        if not app_memories:
            return f"No previous perturbations for {target_app}"

        context_parts = []
        for mem in app_memories[-3:]:  # Last 3 perturbations
            context_parts.append(f"""
            Step {mem["step_idx"]}: {mem["current_action"]}
            - Perturbation Types: {mem["perturbation_types"]}
            - Previous Perturbations: {mem["previous_perturbations"]}
            """)

        return "Recent Perturbation History:\n" + "\n".join(context_parts)

    def _format_app_states_for_decision(self, app_states: List[Dict[str, Any]]) -> str:
        """Format app states for perturbation decision"""
        if not app_states:
            return "No app states available"

        formatted = []
        for app_state in app_states:
            app_name = app_state.get("app_name", "Unknown")
            elements = app_state.get("elements", [])

            # Focus on interactive elements for perturbation decisions
            interactive_elements = [
                elem
                for elem in elements
                if elem.get("element_type", "").lower()
                in ["button", "link", "input", "menu", "checkbox", "radio"]
            ]

            formatted.append(f"App: {app_name} ({len(interactive_elements)} interactive elements)")

        return "\n".join(formatted)

    def _enhance_with_procedural_memory(
        self, llm_decision: Dict[str, Any], scenario_spec: ScenarioSpec, execution_context: ExecutionContext
    ) -> Dict[str, Any]:
        """Enhance LLM decision with procedural memory and autoglm_v capabilities"""
        try:
            target_app = scenario_spec.target_app.lower()
            perturbation_type = llm_decision.get("perturbation_type", "theme")
            parameters = llm_decision.get("parameters", {})

            # Generate coherent perturbation using procedural memory
            coherent_command = self._generate_coherent_perturbation(
                target_app, perturbation_type, parameters, execution_context
            )

            if coherent_command:
                llm_decision["generated_code"] = coherent_command
                llm_decision["api_call"] = self._determine_api_call(target_app, coherent_command)
                llm_decision["procedural_memory_enhanced"] = True
                llm_decision["reasoning"] += " (Enhanced with procedural memory)"

                # Update perturbation history
                self.perturbation_history.append(
                    {
                        "step_idx": execution_context.step_idx,
                        "target_app": target_app,
                        "perturbation_type": perturbation_type,
                        "command": coherent_command,
                        "timestamp": execution_context.timestamp,
                    }
                )

            return llm_decision

        except Exception as e:
            self.logger.error(f"Error enhancing with procedural memory: {e}")
            return llm_decision

    def _generate_coherent_perturbation(
        self,
        target_app: str,
        perturbation_type: str,
        parameters: Dict[str, Any],
        execution_context: ExecutionContext,
    ) -> str:
        """Generate coherent perturbation considering procedural memory"""
        try:
            # Check for recent perturbations to maintain coherence
            recent_perturbations = self.perturbation_history[-2:] if self.perturbation_history else []

            # Generate base perturbation command
            base_command = self.autoglm_generator.generate_perturbation_command(
                target_app, perturbation_type, parameters
            )

            # Enhance with procedural memory context
            if recent_perturbations:
                # Ensure coherence with recent perturbations
                coherent_command = self._ensure_perturbation_coherence(
                    base_command, recent_perturbations, target_app
                )
                return coherent_command

            return base_command

        except Exception as e:
            self.logger.error(f"Error generating coherent perturbation: {e}")
            return ""

    def _ensure_perturbation_coherence(
        self, base_command: str, recent_perturbations: List[Dict[str, Any]], target_app: str
    ) -> str:
        """Ensure perturbation coherence with recent perturbations"""
        try:
            # Check if recent perturbations were on the same app
            same_app_perturbations = [p for p in recent_perturbations if p["target_app"] == target_app]

            if same_app_perturbations:
                # Modify command to be coherent with recent perturbations
                coherent_command = f"""
# Coherent perturbation for {target_app}
# Previous perturbations: {len(same_app_perturbations)} recent changes
{base_command}

# Ensure coherence with recent changes
# Add subtle variation to avoid repetition
"""
                return coherent_command.strip()

            return base_command

        except Exception as e:
            self.logger.error(f"Error ensuring perturbation coherence: {e}")
            return base_command

    def _determine_api_call(self, target_app: str, generated_code: str) -> str:
        """Determine appropriate API call based on target app and generated code"""
        if target_app in ["libreoffice_calc", "libreoffice_writer", "libreoffice_impress"]:
            return "execute_uno_command"
        elif target_app == "chrome":
            return "execute_js_on_page"
        elif target_app == "code":
            return "execute_python_command"
        elif target_app == "vlc":
            return "execute_python_command"
        else:
            return "execute_bash_command"


class CleanElementIdentificationLLM(CleanLLM):
    """Clean element identification using LLM"""

    def identify_target_element(
        self, action_str: str, app_states: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Use LLM to identify the target element from action string and app states"""

        # Convert app states to a simple format for LLM
        app_states_summary = self._format_app_states_for_llm(app_states)

        prompt = f"""
        Find the UI element that this action is trying to interact with.

        Action: "{action_str}"

        Available Elements:
        {app_states_summary}

        Return JSON with just the element identifiers:
        {{
            "name": "element_name",
            "element_type": "element_type",
            "app_name": "app_name",
            "confidence": 0.95,
            "reasoning": "reasoning"
        }}
        """

        response = self.call_llm(prompt)
        result = self.extract_json(response)

        if isinstance(result, list) and len(result) > 0:
            result = result[0]

        if not isinstance(result, dict):
            return None

        # Validate that we found a valid element
        if result.get("name") is None:
            return None

        # Validate required fields
        required_fields = ["name", "element_type", "app_name"]
        if not all(field in result for field in required_fields):
            return None

        return result

    def _format_app_states_for_llm(self, app_states: List[Dict[str, Any]]) -> str:
        """Format app states in a simple way for LLM consumption with element type prioritization"""
        if not app_states:
            return "No app states available"

        formatted_states = []

        for app_state in app_states:
            app_name = app_state.get("app_name", "Unknown")
            elements = app_state.get("elements", [])

            if not elements:
                continue

            app_summary = f"App: {app_name}\n"

            # Group elements by name to show different types for same text
            elements_by_name = {}
            for element in elements:
                element_type = element.get("element_type", "unknown")
                name = element.get("name", "")
                text = element.get("text", "")

                # Format name/text
                display_name = name if name else text
                if not display_name:
                    display_name = f"{element_type}"

                if display_name not in elements_by_name:
                    elements_by_name[display_name] = []
                elements_by_name[display_name].append(element_type)

            # Format grouped elements
            for display_name, element_types in elements_by_name.items():
                if len(element_types) == 1:
                    app_summary += f"  - {display_name} ({element_types[0]})\n"
                else:
                    # Multiple types for same name - show all but prioritize interactive ones
                    interactive_types = [
                        t for t in element_types if t in ["check-box", "button", "menu-item", "combo-box"]
                    ]
                    if interactive_types:
                        # Show interactive types first
                        app_summary += f"  - {display_name} ({', '.join(interactive_types)})\n"
                    else:
                        app_summary += f"  - {display_name} ({', '.join(element_types)})\n"

            formatted_states.append(app_summary)

        return "\n".join(formatted_states)


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

        # Required fields
        required_fields = [
            "scenario_id",
            "target_app",
            "perturbation_trigger",
            "available_perturbation_actions",
            "learning_objectives",
            "target_components",
            "perturbation_types",
        ]

        for field in required_fields:
            if field not in scenario_data:
                errors.append(f"Missing required field: {field}")
            elif not scenario_data[field]:
                errors.append(f"Empty required field: {field}")

        # Validate scenario_id format
        if "scenario_id" in scenario_data:
            scenario_id = scenario_data["scenario_id"]
            if not isinstance(scenario_id, str) or len(scenario_id) < 3:
                errors.append("Invalid scenario_id: must be non-empty string")

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
                valid_types = [pt.value for pt in PerturbationType]
                for pt in perturbation_types:
                    if pt not in valid_types:
                        errors.append(f"Invalid perturbation type: {pt}")

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

        return len(errors) == 0, errors

    def verify_perturbation_decision(self, decision_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Verify perturbation decision completeness and validity"""
        errors = []

        # Required fields
        required_fields = ["should_apply", "reasoning", "perturbation_type", "parameters"]

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
            valid_types = [pt.value for pt in PerturbationType]
            if perturbation_type not in valid_types:
                errors.append(f"Invalid perturbation_type: {perturbation_type}")

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

        # Add default values for missing fields
        defaults = {
            "scenario_id": f"scenario_{hash(str(scenario_data))}",
            "target_app": "system",
            "perturbation_trigger": "During task execution",
            "available_perturbation_actions": 'echo "Default perturbation applied"',
            "learning_objectives": "Learn to adapt to visual changes",
            "target_components": ["general"],
            "perturbation_types": ["theme"],
            "maintains_functionality": True,
            "perturbation_intensity": "medium",
            "realistic_scenario": "Generic perturbation scenario",
        }

        for key, default_value in defaults.items():
            if key not in enhanced or not enhanced[key]:
                enhanced[key] = default_value

        return enhanced
