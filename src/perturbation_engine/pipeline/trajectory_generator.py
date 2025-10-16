"""
TrajectoryGenerator: Single trajectory execution
Seed + spec → perturbed trajectory
"""

import datetime
import json
import logging
import os
import re
import signal
import sys
import time
from typing import Any, Dict

from perturbation_engine.configure_logging import set_run_context
from perturbation_engine.pipeline.app_state_utils import get_timestamp, map_app_name_to_type
from perturbation_engine.pipeline.data_models import (
    ExecutionContext,
    GeneratedTrajectory,
    ScenarioSpec,
    SeedTrajectory,
)
from perturbation_engine.pipeline.llm_services import PerturbationGenerator
from perturbation_engine.pipeline.perturbation_desktop_env import PerturbationDesktopEnv
from perturbation_engine.pipeline.phase_data_manager import PhaseDataManager
from perturbation_engine.pipeline.trajectory_replayer import TrajectoryReplayer
from perturbation_engine.tools.app_state_manager import ElementTracker
from perturbation_engine.utils.memory_utils import force_garbage_collection, log_memory_usage

# TEMP DEBUG: Load intermediate data to bypass LLM calls
TEMP_DEBUG_MODE = True  # Set to False to disable temp debug mode


def _load_temp_debug_data(step_idx: int, trajectory_id: str) -> Dict[str, Any]:
    """Load intermediate data from phases folder for debugging"""
    debug_base = "/Users/lockewang/FIG/software-control/debug"
    phases_dir = os.path.join(
        debug_base,
        trajectory_id.split("_scenario_")[0],
        f"{trajectory_id}_scenario_1_code",
        "run_20251014_183850",
        "phases",
    )

    if not os.path.exists(phases_dir):
        return {}

    result = {}

    # Load perturbation decision
    decision_files = [
        f for f in os.listdir(phases_dir) if f.startswith(f"step_{step_idx:03d}_perturbation_decision")
    ]
    if decision_files:
        decision_path = os.path.join(phases_dir, decision_files[0])
        try:
            with open(decision_path, "r") as f:
                result["perturbation_decision"] = json.load(f)
        except Exception as e:
            print(f"Failed to load decision file: {e}")

    # Load target element
    element_files = [f for f in os.listdir(phases_dir) if f.startswith(f"step_{step_idx:03d}_target_element")]
    if element_files:
        element_path = os.path.join(phases_dir, element_files[0])
        try:
            with open(element_path, "r") as f:
                result["target_element"] = json.load(f)
        except Exception as e:
            print(f"Failed to load element file: {e}")

    return result


class PathManager:
    """Centralized path management for trajectory generation with improved organization"""

    def __init__(self, result_base_dir: str = "/opt/manifold/results", run_id: str = None):
        self.result_base_dir = result_base_dir
        self.run_id = run_id or self._generate_run_id()
        self._ensure_base_directory()

    def _generate_run_id(self) -> str:
        """Generate a unique run ID for this execution"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"run_{timestamp}"

    def _ensure_base_directory(self):
        """Ensure the base result directory exists"""
        os.makedirs(self.result_base_dir, exist_ok=True)

    def get_trajectory_directory(self, trajectory_id: str) -> str:
        """Get the directory path for a specific trajectory with run organization"""
        # Extract seed ID from trajectory ID (e.g., "seed_id_scenario_1_app" -> "seed_id")
        seed_id = trajectory_id.split("_scenario_")[0] if "_scenario_" in trajectory_id else trajectory_id

        # Create organized structure: results/seed_id/run_id/scenario_id/
        return os.path.join(self.result_base_dir, seed_id, self.run_id, trajectory_id)

    def get_trajectory_file_path(self, trajectory_id: str) -> str:
        """Get the trajectory JSONL file path"""
        return os.path.join(self.get_trajectory_directory(trajectory_id), "traj.jsonl")

    def get_recording_path(self, trajectory_id: str) -> str:
        """Get the recording video file path"""
        return os.path.join(self.get_trajectory_directory(trajectory_id), "recording.mp4")

    def get_screenshot_path(self, trajectory_id: str, step_num: int, timestamp: str) -> str:
        """Get the screenshot file path for a specific step"""
        return os.path.join(
            self.get_trajectory_directory(trajectory_id),
            "screenshots",
            f"step_{step_num:03d}_{timestamp}.png",
        )

    def ensure_trajectory_directory(self, trajectory_id: str) -> str:
        """Ensure trajectory directory exists and return its path"""
        trajectory_dir = self.get_trajectory_directory(trajectory_id)
        os.makedirs(trajectory_dir, exist_ok=True)

        # Create subdirectories for better organization
        screenshots_dir = os.path.join(trajectory_dir, "screenshots")
        os.makedirs(screenshots_dir, exist_ok=True)

        return trajectory_dir

    def get_run_summary_path(self, seed_id: str) -> str:
        """Get the path for run summary file"""
        return os.path.join(self.result_base_dir, seed_id, self.run_id, "run_summary.json")

    def get_seed_summary_path(self, seed_id: str) -> str:
        """Get the path for seed summary file"""
        return os.path.join(self.result_base_dir, seed_id, "seed_summary.json")


class TrajectoryGenerator:
    """Execute single trajectory with perturbation injection"""

    def __init__(self, result_base_dir: str = "/opt/manifold/results", run_id: str = None):
        # Ensure logging is configured for subprocess (only if not already configured)
        if not logging.getLogger().handlers:
            from perturbation_engine.configure_logging import configure_logging

            configure_logging()

        self.logger = logging.getLogger(__name__)
        self.perturbation_generator = PerturbationGenerator()
        self.path_manager = PathManager(result_base_dir, run_id)

        self.element_tracker = ElementTracker()

        # Initialize phase data manager for debugging with run_id
        self.phase_data_manager = None  # Will be initialized per trajectory

        log_memory_usage("TrajectoryGenerator initialized", self.logger)

        # Track applied perturbation commands for runtime diversity checking
        self._applied_command_signatures = set()

        # Store current environment for signal handling
        self._current_env = None
        self._current_trajectory_id = None

        # Set up signal handler for graceful shutdown
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown"""

        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, attempting to save recording...")
            self._save_recording_on_interrupt()
            sys.exit(0)

        # Handle SIGINT (Ctrl+C) and SIGTERM
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _save_recording_on_interrupt(self):
        """Save recording when process is interrupted"""
        if self._current_env and self._current_trajectory_id:
            try:
                recording_path = self.path_manager.get_recording_path(self._current_trajectory_id)
                self._current_env.controller.end_recording(recording_path)
                self.logger.info(f"Recording saved on interrupt: {recording_path}")
            except Exception as e:
                self.logger.warning(f"Failed to save recording on interrupt: {e}")

    def execute_trajectory(
        self,
        env: PerturbationDesktopEnv,
        seed_trajectory: SeedTrajectory,
        scenario_spec: ScenarioSpec,
        max_steps: int = 15,
    ) -> GeneratedTrajectory:
        """Execute trajectory with runtime perturbation and error handling"""

        start_time = time.time()
        trajectory_id = f"{scenario_spec.scenario_id}"

        # Store current state for signal handling
        self._current_env = env
        self._current_trajectory_id = trajectory_id

        self.phase_data_manager = PhaseDataManager(trajectory_id, run_id=self.path_manager.run_id)
        self.logger.info(f"Executing trajectory {trajectory_id}: {seed_trajectory.task_instruction}")

        # Set run context for logging
        set_run_context(trajectory_id, self.path_manager.run_id)

        # Track perturbation success rates
        perturbation_attempts = 0
        perturbation_successes = 0
        perturbation_failures = 0

        try:
            trajectory_replayer = TrajectoryReplayer()
            trajectory_replayer.load_trajectory(seed_trajectory.gt_actions_file_path)

            env.reset(task_config=seed_trajectory.config)
            env.controller.start_recording()

            window_states = env.controller.get_window_states()
            done = False
            step_idx = 0
            perturbation_log = []
            action_history = []
            step_by_step_log = []

            # Main execution loop
            while not done and step_idx < trajectory_replayer.get_total_steps():
                cot_response, action = trajectory_replayer.step()

                # Check if trajectory is complete
                if not action:
                    self.logger.info(f"Trajectory completed at step {step_idx}")
                    break

                (
                    step_log_entry,
                    perturbation_attempts,
                    perturbation_successes,
                    perturbation_failures,
                    window_states,
                    done,
                    obs,
                ) = self._execute_single_step(
                    env,
                    step_idx,
                    action,
                    cot_response,
                    window_states,
                    action_history,
                    seed_trajectory,
                    scenario_spec,
                    perturbation_attempts,
                    perturbation_successes,
                    perturbation_failures,
                    trajectory_id,
                    trajectory_replayer.get_total_steps(),
                )

                step_by_step_log.append(step_log_entry)

                if step_log_entry["perturbation_commands"]:
                    perturbation_log.append(
                        {
                            "step": step_idx + 1,
                            "timestamp": step_log_entry["timestamp"],
                            "decision": step_log_entry["perturbation_decision"],
                            "result": {
                                "success": step_log_entry["perturbation_success"],
                                "error_message": step_log_entry["perturbation_failure_reason"] or "",
                            },
                        }
                    )

                if done:
                    self.logger.info("Episode completed")
                    break

                step_idx += 1

            try:
                result = env.evaluate()
                self.logger.info(f"Trajectory evaluation completed successfully: {result}")
            except Exception as e:
                self.logger.error(f"Trajectory evaluation failed: {e}")
                result = 0.0

            generation_time = time.time() - start_time

            env.controller.end_recording(self.path_manager.get_recording_path(trajectory_id))

            perturbation_success_rate = (
                (perturbation_successes / perturbation_attempts) if perturbation_attempts > 0 else 0.0
            )

            self.logger.info(
                f"Perturbation stats for {trajectory_id}: {perturbation_successes}/{perturbation_attempts} successful ({perturbation_success_rate:.2%})"
            )

            perturbation_log.append(
                {
                    "summary": {
                        "perturbation_attempts": perturbation_attempts,
                        "perturbation_successes": perturbation_successes,
                        "perturbation_failures": perturbation_failures,
                        "perturbation_success_rate": perturbation_success_rate,
                    }
                }
            )

            generated_trajectory = self._create_generated_trajectory(
                trajectory_id,
                seed_trajectory,
                scenario_spec,
                result,
                generation_time,
                perturbation_log,
                step_by_step_log,
                perturbation_attempts,
                perturbation_successes,
                env.controller.get_window_states(),
            )

            self.logger.info(
                f"Trajectory {trajectory_id} completed: success={result > 0}, score={result}, perturbation_rate={perturbation_success_rate:.2%}"
            )

            self._cleanup_resources()
            # Clear current state
            self._current_env = None
            self._current_trajectory_id = None
            return generated_trajectory

        except Exception as e:
            # Use repr to avoid format string issues if exception message contains braces
            self.logger.exception(f"Error executing trajectory {trajectory_id}: {repr(e)}")
            # Clean up resources even on failure
            self._cleanup_resources()
            # Clear current state
            self._current_env = None
            self._current_trajectory_id = None
            return self._create_generated_trajectory(
                trajectory_id,
                seed_trajectory,
                scenario_spec,
                0.0,
                time.time() - start_time,
                [],
                [],
                0,
                0,
                None,
            )

    def _cleanup_resources(self):
        """Clean up resources to prevent memory leaks"""
        force_garbage_collection(self.logger)

    def _serialize_scenario_spec(self, scenario_spec: ScenarioSpec) -> Dict[str, Any]:
        """Serialize scenario spec to dictionary format"""
        return {
            "scenario_id": scenario_spec.scenario_id,
            "target_app": scenario_spec.target_app,
            "perturbation_trigger": scenario_spec.perturbation_trigger,
            "available_perturbation_actions": scenario_spec.available_perturbation_actions,
            "learning_objectives": scenario_spec.learning_objectives,
            "target_components": scenario_spec.target_components,
            "perturbation_types": [pt.value for pt in scenario_spec.perturbation_types],
            "perturbation_category": scenario_spec.perturbation_category.value,
        }

    def _extract_command_signature(self, generated_code: str) -> str:
        """
        Extract a semantic signature from generated perturbation code for diversity checking.

        This creates a signature that captures the semantic intent and target elements
        while normalizing specific values, making it possible to detect truly duplicate perturbations.
        """
        if not generated_code:
            return ""

        # Start with the raw command
        signature = generated_code.strip()

        # Extract semantic components for diversity checking
        semantic_parts = []

        # Extract API call type
        api_match = re.search(r"execute_(\w+)_command", signature)
        if api_match:
            semantic_parts.append(f"api:{api_match.group(1)}")

        # Extract target app/component information
        target_app_match = re.search(r"'target_app':\s*['\"]([^'\"]*)['\"]", signature)
        if target_app_match:
            semantic_parts.append(f"app:{target_app_match.group(1)}")

        # Extract visual modification intent
        visual_intent = self._extract_visual_intent(signature)
        if visual_intent:
            semantic_parts.append(f"visual:{visual_intent}")

        # Extract target element/component
        element_target = self._extract_element_target(signature)
        if element_target:
            semantic_parts.append(f"element:{element_target}")

        # Extract perturbation type from command content
        perturbation_type = self._extract_perturbation_type(signature)
        if perturbation_type:
            semantic_parts.append(f"type:{perturbation_type}")

        # If no semantic parts found, fall back to basic normalization
        if not semantic_parts:
            signature = re.sub(r"'([^']*)'", r'"\1"', signature)
            signature = re.sub(r"\s+", " ", signature).strip()
            return signature[:100]  # Truncate to avoid overly long signatures

        return "|".join(semantic_parts)

    def _extract_visual_intent(self, command: str) -> str:
        """Extract the visual modification intent from command"""
        command_lower = command.lower()

        # Theme-related modifications
        if any(theme_word in command_lower for theme_word in ["theme", "gtk-theme", "qt-theme"]):
            return "theme"

        # Color modifications
        if any(color_word in command_lower for color_word in ["color", "background", "border", "rgba", "#"]):
            return "color"

        # Font/typography modifications
        if any(font_word in command_lower for font_word in ["font", "text", "typography", "size"]):
            return "typography"

        # Layout modifications
        if any(
            layout_word in command_lower
            for layout_word in ["margin", "padding", "spacing", "position", "size"]
        ):
            return "layout"

        # CSS/visual styling
        if any(css_word in command_lower for css_word in ["css", "style", "inject", "modify"]):
            return "styling"

        # System-level changes
        if any(sys_word in command_lower for sys_word in ["wallpaper", "desktop", "system", "gsettings"]):
            return "system"

        return ""

    def _extract_element_target(self, command: str) -> str:
        """Extract the target element/component from command"""
        command_lower = command.lower()

        # Common UI element targets
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

        # Check for specific selectors
        if "body" in command_lower:
            return "body"
        elif "html" in command_lower:
            return "html"
        elif "document" in command_lower:
            return "document"

        return ""

    def _extract_perturbation_type(self, command: str) -> str:
        """Extract the perturbation type from command content"""
        command_lower = command.lower()

        # Check for specific perturbation patterns
        if "notify-send" in command_lower:
            return "notification"
        elif "gsettings" in command_lower:
            return "settings"
        elif "inject" in command_lower:
            return "injection"
        elif "modify" in command_lower:
            return "modification"
        elif "change" in command_lower:
            return "change"
        elif "set" in command_lower:
            return "set"

        return ""

    def _is_command_duplicate(self, generated_code: str) -> bool:
        """
        Check if this perturbation command is too similar to previously applied ones.

        Uses semantic analysis to detect meaningful duplicates rather than just API call types.
        Returns True if duplicate (should reject), False if novel (should apply).
        """
        command_sig = self._extract_command_signature(generated_code)

        if not command_sig:
            return False

        # Check for exact semantic signature match
        if command_sig in self._applied_command_signatures:
            self.logger.warning(f"Duplicate perturbation command detected: {command_sig}")
            return True

        # Check for semantic similarity using component analysis
        if self._is_semantically_similar(command_sig):
            self.logger.warning(f"Semantically similar perturbation command detected: {command_sig}")
            return True

        return False

    def _is_semantically_similar(self, command_sig: str) -> bool:
        """
        Check if the command signature is semantically similar to previously applied commands.

        This prevents applying perturbations that are too similar in intent even if not identical.
        """
        if not command_sig or not self._applied_command_signatures:
            return False

        # Parse the command signature components
        current_components = self._parse_signature_components(command_sig)

        # Check against each previously applied command
        for applied_sig in self._applied_command_signatures:
            applied_components = self._parse_signature_components(applied_sig)

            # Calculate similarity score
            similarity_score = self._calculate_similarity_score(current_components, applied_components)

            # If similarity is too high, consider it a duplicate
            if similarity_score >= 0.8:  # 80% similarity threshold
                return True

        return False

    def _parse_signature_components(self, signature: str) -> Dict[str, str]:
        """Parse signature into component dictionary"""
        components = {}

        if "|" in signature:
            parts = signature.split("|")
            for part in parts:
                if ":" in part:
                    key, value = part.split(":", 1)
                    components[key] = value
        else:
            # Fallback for non-semantic signatures
            components["raw"] = signature

        return components

    def _calculate_similarity_score(self, current: Dict[str, str], applied: Dict[str, str]) -> float:
        """
        Calculate similarity score between two command signatures.

        Returns a score between 0.0 (completely different) and 1.0 (identical).
        """
        if not current or not applied:
            return 0.0

        # Weight different components differently
        weights = {
            "api": 0.2,  # API type is less important for diversity
            "app": 0.3,  # Target app is moderately important
            "visual": 0.4,  # Visual intent is very important for diversity
            "element": 0.3,  # Target element is moderately important
            "type": 0.2,  # Perturbation type is less important
            "raw": 0.1,  # Raw signature is least important
        }

        total_weight = 0.0
        weighted_score = 0.0

        # Check each component
        for component, weight in weights.items():
            if component in current and component in applied:
                if current[component] == applied[component]:
                    weighted_score += weight
                total_weight += weight
            elif component in current or component in applied:
                # One has the component, the other doesn't - partial similarity
                weighted_score += weight * 0.5
                total_weight += weight

        # Normalize by total weight
        if total_weight > 0:
            return weighted_score / total_weight

        return 0.0

    def _record_applied_command(self, generated_code: str):
        """Record a successfully applied perturbation command for diversity tracking."""
        command_sig = self._extract_command_signature(generated_code)
        self._applied_command_signatures.add(command_sig)
        self.logger.debug(
            f"Recorded perturbation command signature ({len(self._applied_command_signatures)} total): {command_sig[:80]}..."
        )

    def _create_perturbation_command(
        self, perturbation_decision: Dict[str, Any], perturbation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create perturbation command dictionary from decision and result"""
        return {
            "perturbation_type": perturbation_decision.get("perturbation_type", "unknown"),
            "target_app": perturbation_decision.get("target_app", "unknown"),
            "api_call": perturbation_decision.get("api_call", "unknown"),
            "generated_code": perturbation_decision.get("generated_command", ""),
            "parameters": perturbation_decision.get("parameters", {}),
            "reasoning": perturbation_decision.get("reasoning", ""),
            "success": perturbation_result.get("success", False),
            "error_message": perturbation_result.get("error_message", ""),
            "operation_type": perturbation_result.get("operation_type", ""),
        }

    def _create_step_log_entry(
        self,
        step_idx: int,
        timestamp: str,
        task_instruction: str,
        window_states: list,
        action: Any,
        perturbation_decision: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create step log entry with common fields"""
        return {
            "step": step_idx,
            "timestamp": timestamp,
            "task_instruction": task_instruction,
            "window_states": [
                {
                    "window_id": window_state.window_id,
                    "window_name": window_state.window_name,
                    "app_name": window_state.app_name,
                    "is_active": window_state.is_active,
                    "is_modal": window_state.is_modal,
                    "is_minimized": window_state.is_minimized,
                    "geometry": window_state.geometry,
                    "z_order": window_state.z_order,
                    "x11_window_id": window_state.x11_window_id,
                    "is_mapped": window_state.is_mapped,
                    "desktop": window_state.desktop,
                    "root_element": window_state.root_element.to_dict()
                    if window_state.root_element
                    else None,
                }
                for window_state in window_states
            ]
            if window_states
            else [],
            "action": str(action),
            "perturbation_commands": [],
            "perturbation_success": False,
            "perturbation_failure_reason": None,
            "perturbation_decision": perturbation_decision,
        }

    def _create_generated_trajectory(
        self,
        trajectory_id: str,
        seed_trajectory: SeedTrajectory,
        scenario_spec: ScenarioSpec,
        result: float,
        generation_time: float,
        perturbation_log: list,
        step_by_step_log: list,
        perturbation_attempts: int,
        perturbation_successes: int,
        final_window_states: list = None,
    ) -> GeneratedTrajectory:
        """Create GeneratedTrajectory with all required fields"""
        # Extract successful and failed perturbation commands
        successful_commands = []
        failed_commands = []
        for step in step_by_step_log:
            for command in step.get("perturbation_commands", []):
                if command.get("success", False):
                    successful_commands.append(command)
                else:
                    failed_commands.append(command)

        return GeneratedTrajectory(
            trajectory_id=trajectory_id,
            seed_trajectory_id=seed_trajectory.task_id,
            scenario_spec_id=scenario_spec.scenario_id,
            success=result > 0,
            quality_score=result,
            generation_time=generation_time,
            trajectory_file_path=self.path_manager.get_trajectory_file_path(trajectory_id),
            perturbation_log=perturbation_log,
            scenario_spec_content=self._serialize_scenario_spec(scenario_spec),
            final_app_states=final_window_states,
            total_perturbation_attempts=perturbation_attempts,
            total_perturbation_successes=perturbation_successes,
            step_by_step_log=step_by_step_log,
            successful_perturbation_commands=successful_commands,
            failed_perturbation_commands=failed_commands,
        )

    def _execute_single_step(
        self,
        env: PerturbationDesktopEnv,
        step_idx: int,
        action: str,
        cot_response: str,
        window_states: list,
        action_history: list,
        seed_trajectory: SeedTrajectory,
        scenario_spec: ScenarioSpec,
        perturbation_attempts: int,
        perturbation_successes: int,
        perturbation_failures: int,
        trajectory_id: str,
        total_steps: int,
    ) -> tuple[Dict[str, Any], int, int, int, list, bool, Dict[str, Any]]:
        """Execute a single step with robust perturbation and coordinate tracking."""
        action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
        action_str = str(action)

        self.logger.info(f"Step {step_idx + 1}: {action_str[:100]}")

        # ========== Phase 1: Identify Target Element Candidates (BEFORE Perturbation) ==========
        # TEMP DEBUG: Load target element from phases folder instead of calling LLM
        temp_debug_data = _load_temp_debug_data(step_idx, trajectory_id) if TEMP_DEBUG_MODE else {}
        if TEMP_DEBUG_MODE and temp_debug_data.get("target_element"):
            # Convert dict back to UIElement-like object for compatibility
            element_data = temp_debug_data["target_element"]
            from perturbation_engine.pipeline.data_models import UIElement, VisibilityState

            target_element = UIElement(
                element_id=element_data["element_id"],
                element_type=element_data["element_type"],
                name=element_data["name"],
                position=element_data["position"],
                parent_id=element_data.get("parent_id"),
                depth=element_data.get("depth", 0),
                visibility=VisibilityState.VISIBLE,
                is_enabled=element_data.get("is_enabled", True),
                is_focused=element_data.get("is_focused", False),
                is_expanded=element_data.get("is_expanded", False),
                properties=element_data.get("properties", {}),
                children=[],
            )
            self.logger.info(f"TEMP DEBUG: Loaded target element from phases folder for step {step_idx}")
        else:
            # Retry 3 times if no target element candidates are found
            for _ in range(3):
                target_element_candidates = self.element_tracker.identify_target_element_candidates(
                    action_str, window_states
                )
                if len(target_element_candidates) == 0:
                    self.logger.warning("✗ No target element candidates found")
                else:
                    break

            target_element = target_element_candidates[0] if target_element_candidates else None

        # Create element visualization for debugging
        try:
            screenshot_data = None
            try:
                obs_temp = env.controller.get_screenshot()
                if obs_temp:
                    screenshot_data = obs_temp
            except Exception as e:
                self.logger.warning(f"Could not get screenshot for visualization: {e}")

            # Use PhaseDataManager for visualization (saves in debug folder structure)
            visualization_path = self.phase_data_manager.visualize_element_bounding_boxes(
                window_states,
                target_element_id=target_element.element_id if target_element else None,
                screenshot_data=screenshot_data,
                step_idx=step_idx,
            )
            if visualization_path:
                self.logger.info(f"Element visualization saved: {visualization_path}")
        except Exception as e:
            self.logger.exception(f"Could not create element visualization: {e}")

        # Save Phase 1 data
        if target_element:
            self.phase_data_manager.save_element_identity(step_idx, target_element)
            self.logger.info(
                f"✓ Target identified: {target_element.element_id} "
                f"'{target_element.name[:20] if target_element.name else 'unnamed'}' "
                f"at ({target_element.position['center_x']}, {target_element.position['center_y']}) "
            )
        else:
            self.logger.error(
                f"✗ Failed to identify valid target element from {len(target_element_candidates)} candidates"
            )

        # Save window states using phase data manager
        self.phase_data_manager.save_window_states(step_idx, "before_perturbation", window_states)

        # ========== Phase 2: Perturbation Decision ==========
        execution_context = ExecutionContext(
            step_idx=step_idx,
            current_action=action_str,
            action_history=action_history.copy(),
            cot_context=cot_response,
            window_states=window_states,  # Pass window_states directly
            task_instruction=seed_trajectory.task_instruction,
            task_type=seed_trajectory.task_type,
            scenario_spec=scenario_spec,
            total_steps=total_steps,  # Pass total steps for strategic timing
        )

        # Save Phase 2 data
        self.phase_data_manager.save_execution_context(step_idx, execution_context)

        # TEMP DEBUG: Load perturbation decision from phases folder instead of calling LLM
        temp_debug_data = _load_temp_debug_data(step_idx, trajectory_id) if TEMP_DEBUG_MODE else {}
        if TEMP_DEBUG_MODE and temp_debug_data.get("perturbation_decision"):
            perturbation_decision = temp_debug_data["perturbation_decision"]
            self.logger.info(
                f"TEMP DEBUG: Loaded perturbation decision from phases folder for step {step_idx}"
                f"TEMP DEBUG: Loaded perturbation decision from phases folder for step {step_idx}"
            )
        else:
            perturbation_decision = self.perturbation_generator.decide_perturbation(
                execution_context, scenario_spec
            )

        self.phase_data_manager.save_perturbation_decision(step_idx, perturbation_decision)

        step_log_entry = self._create_step_log_entry(
            step_idx,
            action_timestamp,
            seed_trajectory.task_instruction,
            window_states,
            action,
            perturbation_decision,
        )

        # ========== Phase 3: Apply Perturbation (if decided) ==========
        perturbation_applied = False

        if perturbation_decision.get("should_apply", False):
            # Check for duplicate commands (diversity)
            generated_command = perturbation_decision.get("generated_command", "")

            # if self._is_command_duplicate(generated_command):
            #     self.logger.warning(f"Skipping duplicate perturbation at step {step_idx}")
            #     step_log_entry["perturbation_failure_reason"] = "Duplicate command"
            #     step_log_entry["perturbation_commands"].append(
            #         {
            #             "success": False,
            #             "operation_type": "rejected_duplicate",
            #         }
            #     )
            # else:
            #     perturbation_attempts += 1

            try:
                # Apply perturbation
                perturbation_result = self._apply_perturbation(env.controller, perturbation_decision)

                # Save Phase 3 data with enhanced logging
                self.phase_data_manager.save_perturbation_result(step_idx, perturbation_result)

                # Get window states after perturbation for comprehensive debugging
                window_states_after = env.controller.get_window_states()

                # Save comprehensive perturbation debugging data to debug folder
                self._save_comprehensive_perturbation_debug_data(
                    step_idx,
                    perturbation_decision,
                    perturbation_result,
                    window_states,
                    window_states_after,
                    target_element,
                )

                # Create perturbation command for step log (minimal data)
                perturbation_command = self._create_perturbation_command(
                    perturbation_decision, perturbation_result
                )
                step_log_entry["perturbation_commands"].append(perturbation_command)

                if perturbation_result.get("success", False):
                    perturbation_successes += 1
                    perturbation_applied = True
                    step_log_entry["perturbation_success"] = True

                    env.mark_perturbation_applied()
                    self._record_applied_command(generated_command)

                    self.logger.info(f"Perturbation applied: {perturbation_decision.get('reasoning', '')}")

                else:
                    perturbation_failures += 1
                    step_log_entry["perturbation_success"] = False
                    step_log_entry["perturbation_failure_reason"] = perturbation_result.get(
                        "error_message", "Unknown"
                    )

            except Exception as e:
                perturbation_failures += 1
                step_log_entry["perturbation_success"] = False
                step_log_entry["perturbation_failure_reason"] = str(e)
                self.logger.error(f"Perturbation error: {e}")

        # ========== Phase 4: Update Action Coordinates ==========
        if target_element and perturbation_applied:
            # Get fresh window states after perturbation
            window_states_after = env.controller.get_window_states()

            # Save window states after perturbation using phase data manager
            self.phase_data_manager.save_window_states(step_idx, "after_perturbation", window_states_after)

            # Track element in new states using autoglm_v
            target_element = self.element_tracker.track_element_after_perturbation(
                target_element, window_states_after
            )

        # ========== Phase 5: Execute Action ==========
        self.logger.debug(f"Executing: {action_str[:100]}")
        action = self.element_tracker.update_action_coordinates(action_str, target_element.position)
        obs, reward, done, info = env.step(action)
        action_history.append(action_str)

        if reward < 0:
            self.logger.warning(f"Negative reward: {reward}")
        if done:
            self.logger.info(f"Episode completed at step {step_idx + 1}")

        # Update window states after action
        window_states = env.controller.get_window_states()

        step_log_entry.update(
            {
                "window_states_after_action": [
                    {
                        "window_id": window_state.window_id,
                        "window_name": window_state.window_name,
                        "app_name": window_state.app_name,
                        "is_active": window_state.is_active,
                        "is_modal": window_state.is_modal,
                        "is_minimized": window_state.is_minimized,
                        "geometry": window_state.geometry,
                        "z_order": window_state.z_order,
                        "x11_window_id": window_state.x11_window_id,
                        "is_mapped": window_state.is_mapped,
                        "desktop": window_state.desktop,
                        "root_element": window_state.root_element.to_dict()
                        if window_state.root_element
                        else None,
                    }
                    for window_state in window_states
                ]
                if window_states
                else [],
                "target_element": target_element.to_dict() if target_element else None,
                "reward": reward,
                "done": done,
                "info": info,
            }
        )

        # Save complete step log
        self.phase_data_manager.save_step_log(step_idx, step_log_entry)

        # ========== Phase 6: Save Trajectory Data ==========
        self._save_trajectory_step(
            trajectory_id,
            step_idx + 1,
            action_timestamp,
            action,
            cot_response,
            reward,
            done,
            info,
            obs,
            perturbation_decision.get("should_apply", False),
            task_instruction=seed_trajectory.task_instruction,
            target_element=target_element,
        )

        return (
            step_log_entry,
            perturbation_attempts,
            perturbation_successes,
            perturbation_failures,
            window_states,
            done,
            obs,
        )

    def _apply_perturbation(self, controller, perturbation_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Apply perturbation using enhanced controller"""

        try:
            api_call = perturbation_decision.get("api_call")
            if not api_call:
                raise ValueError("Missing api_call in perturbation decision")

            result = controller.execute_perturbation(
                perturbation_type=perturbation_decision.get("perturbation_type", "unknown"),
                generated_code=perturbation_decision.get("generated_command", ""),
                api_call=api_call,
                parameters=perturbation_decision.get("parameters", {}),
            )

            return {
                "success": result.success,
                "operation_type": result.operation_type,
                "target_app": result.target_app,
                "error_message": result.error_message,
                "method": "clean_perturbation_generator",
                "result_data": result.result_data,  # Include detailed result data
            }

        except Exception as e:
            self.logger.error(f"Error applying perturbation: {e}")
            return {"success": False, "error": str(e)}

    def _save_detailed_perturbation_execution(
        self,
        step_idx: int,
        perturbation_decision: Dict[str, Any],
        perturbation_result: Dict[str, Any],
        window_states_before: list,
    ) -> str:
        """Save detailed perturbation execution data for debugging"""
        try:
            # Get window states after perturbation
            window_states_after = None
            try:
                # This would need access to the controller, but we'll work with what we have
                pass
            except Exception as e:
                self.logger.warning(f"Could not get window states after perturbation: {e}")

            detailed_data = {
                "step_idx": step_idx,
                "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                "perturbation_decision": {
                    "should_apply": perturbation_decision.get("should_apply", False),
                    "reasoning": perturbation_decision.get("reasoning", ""),
                    "perturbation_type": perturbation_decision.get("perturbation_type", ""),
                    "api_call": perturbation_decision.get("api_call", ""),
                    "generated_command": perturbation_decision.get("generated_command", ""),
                    "parameters": perturbation_decision.get("parameters", {}),
                    "confidence": perturbation_decision.get("confidence", 0.0),
                    "visual_impact": perturbation_decision.get("visual_impact", ""),
                },
                "perturbation_result": {
                    "success": perturbation_result.get("success", False),
                    "operation_type": perturbation_result.get("operation_type", ""),
                    "target_app": perturbation_result.get("target_app", ""),
                    "error_message": perturbation_result.get("error_message", ""),
                    "method": perturbation_result.get("method", ""),
                    "result_data": perturbation_result.get("result_data", {}),
                },
                "window_states_before": [self._serialize_window_state(ws) for ws in window_states_before]
                if window_states_before
                else [],
                "window_states_after": window_states_after or [],
                "analysis": {
                    "command_parsed": perturbation_decision.get("generated_command", ""),
                    "api_call_used": perturbation_decision.get("api_call", ""),
                    "execution_success": perturbation_result.get("success", False),
                    "potential_side_effects": self._analyze_potential_side_effects(
                        perturbation_decision, perturbation_result
                    ),
                },
            }

            return self.phase_data_manager.save_phase_data(
                step_idx, "detailed_perturbation_execution", detailed_data
            )

        except Exception as e:
            self.logger.error(f"Error saving detailed perturbation execution: {e}")
            return ""

    def _serialize_window_state(self, window_state) -> Dict[str, Any]:
        """Serialize window state for logging"""
        try:
            return {
                "window_id": window_state.window_id,
                "window_name": window_state.window_name,
                "app_name": window_state.app_name,
                "is_active": window_state.is_active,
                "is_modal": window_state.is_modal,
                "is_minimized": window_state.is_minimized,
                "geometry": window_state.geometry,
                "z_order": window_state.z_order,
                "x11_window_id": window_state.x11_window_id,
                "is_mapped": window_state.is_mapped,
                "desktop": window_state.desktop,
                "root_element": self._serialize_element(window_state.root_element)
                if window_state.root_element
                else None,
            }
        except Exception as e:
            self.logger.warning(f"Error serializing window state: {e}")
            return {"error": str(e)}

    def _serialize_element(self, element) -> Dict[str, Any]:
        """Serialize UI element for logging"""
        try:
            return {
                "element_id": element.element_id,
                "element_type": element.element_type,
                "name": element.name,
                "position": element.position,
                "parent_id": element.parent_id,
                "depth": element.depth,
                "visibility": element.visibility.value
                if hasattr(element.visibility, "value")
                else str(element.visibility),
                "is_enabled": element.is_enabled,
                "is_focused": element.is_focused,
                "is_expanded": element.is_expanded,
                "properties": element.properties,
                "children_count": len(element.children) if hasattr(element, "children") else 0,
            }
        except Exception as e:
            self.logger.warning(f"Error serializing element: {e}")
            return {"error": str(e)}

    def _analyze_potential_side_effects(
        self, perturbation_decision: Dict[str, Any], perturbation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze potential side effects of perturbation commands"""
        side_effects = {
            "system_level_changes": [],
            "app_specific_changes": [],
            "ui_state_changes": [],
            "potential_conflicts": [],
        }

        api_call = perturbation_decision.get("api_call", "")
        generated_command = perturbation_decision.get("generated_command", "")
        _target_app = perturbation_decision.get("parameters", {}).get("target_app", "")

        # Analyze gsettings commands
        if "gsettings" in generated_command.lower():
            side_effects["system_level_changes"].append(
                {
                    "type": "desktop_theme_change",
                    "command": generated_command,
                    "impact": "May affect all GTK applications including LibreOffice",
                    "risk_level": "medium",
                    "description": "Desktop theme changes can cause LibreOffice to refresh its UI or show file dialogs",
                }
            )

        # Analyze UNO commands
        if api_call == "execute_uno_command":
            side_effects["app_specific_changes"].append(
                {
                    "type": "libreoffice_internal_change",
                    "command": generated_command,
                    "impact": "Direct LibreOffice internal state modification",
                    "risk_level": "high",
                    "description": "UNO commands can trigger LibreOffice to show dialogs or change state unexpectedly",
                }
            )

        # Analyze CSS injection
        if api_call == "execute_css_injection":
            side_effects["ui_state_changes"].append(
                {
                    "type": "visual_styling_change",
                    "command": generated_command,
                    "impact": "Changes visual appearance of web elements",
                    "risk_level": "low",
                    "description": "CSS changes typically don't affect application state",
                }
            )

        return side_effects

    def _save_trajectory_step(
        self,
        trajectory_id: str,
        step_num: int,
        timestamp: str,
        action: str,
        cot_response: str,
        reward: float,
        done: bool,
        info: Dict[str, Any],
        obs: Dict[str, Any],
        perturbation_applied: bool,
        task_instruction: str = "",
        target_element: Dict[str, Any] = None,
    ):
        """Save individual trajectory step data with full screenshot and metadata"""
        try:
            # Ensure trajectory directory exists
            self.path_manager.ensure_trajectory_directory(trajectory_id)

            # Always save screenshot
            screenshot_saved = False
            if "screenshot" in obs and obs["screenshot"] is not None:
                screenshot_path = self.path_manager.get_screenshot_path(trajectory_id, step_num, timestamp)
                with open(screenshot_path, "wb") as f:
                    f.write(obs["screenshot"])
                screenshot_saved = True
                self.logger.debug(f"Saved screenshot: {screenshot_path}")

            # Save clean trajectory data with only essential execution information
            trajectory_data = {
                "step_num": step_num,
                "action_timestamp": timestamp,
                "action": action,  # Keep original action format
                "reward": reward,
                "done": done,
                "perturbation_applied": perturbation_applied,  # Keep simple boolean flag
            }

            if info:
                trajectory_data["info"] = info

            if cot_response:
                trajectory_data["cot_response"] = cot_response

            if task_instruction:
                trajectory_data["task_instruction"] = task_instruction

            # Keep target element info for action context (but not full window states)
            if target_element:
                trajectory_data["target_element"] = {
                    "element_id": target_element.element_id,
                    "name": target_element.name,
                    "position": target_element.position,
                }

            if screenshot_saved:
                trajectory_data["screenshot_file"] = f"step_{step_num:03d}_{timestamp}.png"

            with open(self.path_manager.get_trajectory_file_path(trajectory_id), "a") as f:
                f.write(json.dumps(trajectory_data))
                f.write("\n")

        except Exception as e:
            self.logger.error(f"Error saving trajectory step: {e}")

    def _save_comprehensive_perturbation_debug_data(
        self,
        step_idx: int,
        perturbation_decision: Dict[str, Any],
        perturbation_result: Dict[str, Any],
        window_states_before: list,
        window_states_after: list = None,
        target_element: Dict[str, Any] = None,
    ):
        """Save comprehensive perturbation debugging data to debug folder"""
        try:
            # Save detailed perturbation execution data
            self._save_detailed_perturbation_execution(
                step_idx, perturbation_decision, perturbation_result, window_states_before
            )

            # Save perturbation command details
            perturbation_command = self._create_perturbation_command(
                perturbation_decision, perturbation_result
            )
            self.phase_data_manager.save_perturbation_command(step_idx, perturbation_command)

            # Save comprehensive perturbation summary
            comprehensive_summary = {
                "step_idx": step_idx,
                "timestamp": datetime.datetime.now().isoformat(),
                "perturbation_decision": {
                    "should_apply": perturbation_decision.get("should_apply", False),
                    "reasoning": perturbation_decision.get("reasoning", ""),
                    "perturbation_type": perturbation_decision.get("perturbation_type", ""),
                    "target_app": perturbation_decision.get("target_app", ""),
                    "api_call": perturbation_decision.get("api_call", ""),
                    "parameters": perturbation_decision.get("parameters", {}),
                    "generated_command": perturbation_decision.get("generated_command", ""),
                    "confidence": perturbation_decision.get("confidence", 0.0),
                    "visual_impact": perturbation_decision.get("visual_impact", ""),
                },
                "perturbation_result": {
                    "success": perturbation_result.get("success", False),
                    "operation_type": perturbation_result.get("operation_type", ""),
                    "target_app": perturbation_result.get("target_app", ""),
                    "error_message": perturbation_result.get("error_message", ""),
                    "method": perturbation_result.get("method", ""),
                    "result_data": perturbation_result.get("result_data", {}),
                },
                "target_element": target_element.to_dict() if target_element else None,
                "window_states_before": [self._serialize_window_state(ws) for ws in window_states_before]
                if window_states_before
                else [],
                "window_states_after": [self._serialize_window_state(ws) for ws in window_states_after]
                if window_states_after
                else [],
                "analysis": {
                    "command_parsed": perturbation_decision.get("generated_command", ""),
                    "api_call_used": perturbation_decision.get("api_call", ""),
                    "execution_success": perturbation_result.get("success", False),
                    "potential_side_effects": self._analyze_potential_side_effects(
                        perturbation_decision, perturbation_result
                    ),
                },
            }

            self.phase_data_manager.save_perturbation_summary(step_idx, comprehensive_summary)

        except Exception as e:
            self.logger.error(f"Error saving comprehensive perturbation debug data: {e}")

    def _map_app_name_to_type(self, app_name: str) -> str:
        """Map application name to app type - delegate to shared utility"""
        return map_app_name_to_type(app_name)

    def _get_timestamp(self) -> str:
        """Get current timestamp - delegate to shared utility"""
        return get_timestamp()
