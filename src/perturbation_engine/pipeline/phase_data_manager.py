"""
Phase Data Manager: Save and reuse intermediate data between phases
Enables isolated debugging and faster iteration
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class ElementIdentity:
    """Simple element identity for tracking"""

    element_id: str
    name: str
    text: str
    position: Dict[str, int]


class PhaseDataManager:
    """Manages intermediate data between trajectory execution phases"""

    def __init__(self, trajectory_id: str, debug_dir: str = "./debug"):
        self.trajectory_id = trajectory_id
        self.debug_dir = debug_dir
        self.logger = logging.getLogger(__name__)
        self._ensure_debug_directory()

    def _ensure_debug_directory(self):
        """Ensure debug directory exists"""
        os.makedirs(self.debug_dir, exist_ok=True)
        os.makedirs(os.path.join(self.debug_dir, self.trajectory_id), exist_ok=True)

    def save_phase_data(self, step_idx: int, phase: str, data: Dict[str, Any]) -> str:
        """Save phase data to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"step_{step_idx:03d}_{phase}_{timestamp}.json"
        filepath = os.path.join(self.debug_dir, self.trajectory_id, filename)

        # Convert non-serializable objects
        serializable_data = self._make_serializable(data)

        with open(filepath, "w") as f:
            json.dump(serializable_data, f, indent=2)

        self.logger.debug(f"Saved {phase} data: {filepath}")
        return filepath

    def load_phase_data(self, step_idx: int, phase: str) -> Optional[Dict[str, Any]]:
        """Load most recent phase data for step"""
        pattern = f"step_{step_idx:03d}_{phase}_"
        debug_dir = os.path.join(self.debug_dir, self.trajectory_id)

        if not os.path.exists(debug_dir):
            return None

        # Find most recent file matching pattern
        files = [f for f in os.listdir(debug_dir) if f.startswith(pattern) and f.endswith(".json")]
        if not files:
            return None

        # Sort by timestamp (newest first)
        files.sort(reverse=True)
        filepath = os.path.join(debug_dir, files[0])

        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            self.logger.debug(f"Loaded {phase} data: {filepath}")
            return data
        except Exception as e:
            self.logger.error(f"Error loading {phase} data: {e}")
            return None

    def save_element_identity(self, step_idx: int, element: ElementIdentity) -> str:
        """Save ElementIdentity with special handling"""
        data = {
            "element_id": element.element_id,
            "name": element.name,
            "text": element.text,
            "position": element.position,
        }
        return self.save_phase_data(step_idx, "target_element", data)

    def load_element_identity(self, step_idx: int) -> Optional[ElementIdentity]:
        """Load ElementIdentity from saved data"""
        data = self.load_phase_data(step_idx, "target_element")
        if not data:
            return None

        try:
            return ElementIdentity(
                element_id=data["element_id"], name=data["name"], text=data["text"], position=data["position"]
            )
        except Exception as e:
            self.logger.error(f"Error reconstructing ElementIdentity: {e}")
            return None

    def save_execution_context(self, step_idx: int, context: Any) -> str:
        """Save ExecutionContext"""
        data = {
            "step_idx": context.step_idx,
            "current_action": context.current_action,
            "action_history": context.action_history,
            "cot_context": context.cot_context,
            "app_states": context.app_states,
            "task_instruction": context.task_instruction,
            "task_type": context.task_type,
            "scenario_spec": {
                "scenario_id": context.scenario_spec.scenario_id,
                "target_app": context.scenario_spec.target_app,
                "perturbation_trigger": context.scenario_spec.perturbation_trigger,
                "available_perturbation_actions": context.scenario_spec.available_perturbation_actions,
                "learning_objectives": context.scenario_spec.learning_objectives,
                "target_components": context.scenario_spec.target_components,
                "perturbation_types": [pt.value for pt in context.scenario_spec.perturbation_types],
            },
        }
        return self.save_phase_data(step_idx, "execution_context", data)

    def save_perturbation_decision(self, step_idx: int, decision: Dict[str, Any]) -> str:
        """Save perturbation decision"""
        return self.save_phase_data(step_idx, "perturbation_decision", decision)

    def save_perturbation_result(self, step_idx: int, result: Dict[str, Any]) -> str:
        """Save perturbation result"""
        return self.save_phase_data(step_idx, "perturbation_result", result)

    def save_app_states(self, step_idx: int, phase: str, app_states: List[Dict[str, Any]]) -> str:
        """Save app states for specific phase"""
        data = {"app_states": app_states, "phase": phase}
        return self.save_phase_data(step_idx, f"app_states_{phase}", data)

    def load_app_states(self, step_idx: int, phase: str) -> Optional[List[Dict[str, Any]]]:
        """Load app states for specific phase"""
        data = self.load_phase_data(step_idx, f"app_states_{phase}")
        return data.get("app_states") if data else None

    def save_action_update(
        self, step_idx: int, original_action: str, updated_action: str, element_movement: Dict[str, Any]
    ) -> str:
        """Save action coordinate update"""
        data = {
            "original_action": original_action,
            "updated_action": updated_action,
            "element_movement": element_movement,
        }
        return self.save_phase_data(step_idx, "action_update", data)

    def save_step_log(self, step_idx: int, step_log: Dict[str, Any]) -> str:
        """Save complete step log"""
        return self.save_phase_data(step_idx, "step_log", step_log)

    def get_phase_summary(self, step_idx: int) -> Dict[str, Any]:
        """Get summary of all phases for a step"""
        debug_dir = os.path.join(self.debug_dir, self.trajectory_id)
        if not os.path.exists(debug_dir):
            return {}

        pattern = f"step_{step_idx:03d}_"
        files = [f for f in os.listdir(debug_dir) if f.startswith(pattern) and f.endswith(".json")]

        phases = {}
        for file in files:
            phase_name = file.replace(f"step_{step_idx:03d}_", "").replace(".json", "")
            timestamp = phase_name.split("_")[-1] if "_" in phase_name else "unknown"
            phase_name = "_".join(phase_name.split("_")[:-1]) if "_" in phase_name else phase_name

            if phase_name not in phases:
                phases[phase_name] = []
            phases[phase_name].append(
                {"file": file, "timestamp": timestamp, "path": os.path.join(debug_dir, file)}
            )

        return phases

    def _make_serializable(self, data: Any) -> Any:
        """Convert data to JSON-serializable format"""
        if isinstance(data, dict):
            return {k: self._make_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._make_serializable(item) for item in data]
        elif isinstance(data, (str, int, float, bool, type(None))):
            return data
        else:
            # Convert other types to string representation
            return str(data)


class PhaseDebugger:
    """Debug individual phases using saved data"""

    def __init__(self, trajectory_id: str, debug_dir: str = "./debug"):
        self.data_manager = PhaseDataManager(trajectory_id, debug_dir)
        self.logger = logging.getLogger(__name__)

    def debug_phase_1_element_identification(
        self, step_idx: int, action: str, app_states: List[Dict[str, Any]]
    ) -> Optional[ElementIdentity]:
        """Debug Phase 1: Element Identification"""
        self.logger.info(f"Debugging Phase 1 for step {step_idx}")

        # Try to load existing data
        existing_element = self.data_manager.load_element_identity(step_idx)
        if existing_element:
            self.logger.info(f"Using cached element: {existing_element.element_id}")
            return existing_element

        # If no cached data, this would normally call the element tracker
        # For debugging, you can manually inspect the data
        self.logger.info("No cached element data found - would need to run element identification")
        return None

    def debug_phase_2_perturbation_decision(self, step_idx: int) -> Optional[Dict[str, Any]]:
        """Debug Phase 2: Perturbation Decision"""
        self.logger.info(f"Debugging Phase 2 for step {step_idx}")

        decision = self.data_manager.load_phase_data(step_idx, "perturbation_decision")
        if decision:
            self.logger.info(f"Found cached decision: should_apply={decision.get('should_apply')}")
            return decision

        self.logger.info("No cached decision data found")
        return None

    def debug_phase_3_perturbation_application(self, step_idx: int) -> Optional[Dict[str, Any]]:
        """Debug Phase 3: Perturbation Application"""
        self.logger.info(f"Debugging Phase 3 for step {step_idx}")

        result = self.data_manager.load_phase_data(step_idx, "perturbation_result")
        if result:
            self.logger.info(f"Found cached result: success={result.get('success')}")
            return result

        self.logger.info("No cached result data found")
        return None

    def debug_phase_4_coordinate_update(self, step_idx: int) -> Optional[Dict[str, Any]]:
        """Debug Phase 4: Coordinate Update"""
        self.logger.info(f"Debugging Phase 4 for step {step_idx}")

        update_data = self.data_manager.load_phase_data(step_idx, "action_update")
        if update_data:
            self.logger.info(f"Found cached update: {update_data.get('element_movement', {})}")
            return update_data

        self.logger.info("No cached update data found")
        return None

    def replay_step_from_phase(self, step_idx: int, start_phase: int) -> Dict[str, Any]:
        """Replay step starting from specific phase"""
        self.logger.info(f"Replaying step {step_idx} from phase {start_phase}")

        summary = self.data_manager.get_phase_summary(step_idx)
        replay_data = {
            "step_idx": step_idx,
            "start_phase": start_phase,
            "available_phases": list(summary.keys()),
            "phase_data": {},
        }

        # Load data for each phase
        for phase_name in summary.keys():
            replay_data["phase_data"][phase_name] = self.data_manager.load_phase_data(step_idx, phase_name)

        return replay_data
