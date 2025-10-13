"""
Phase Data Manager: Save and reuse intermediate data between phases
Enables isolated debugging and faster iteration
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional

from PIL import Image, ImageDraw, ImageFont


@dataclass
class ElementIdentity:
    """Simple element identity for tracking"""

    element_id: str
    name: str
    text: str
    position: Dict[str, int]


class PhaseDataManager:
    """Manages intermediate data between trajectory execution phases"""

    def __init__(self, trajectory_id: str, debug_dir: str = "./debug", run_id: str = None):
        self.trajectory_id = trajectory_id
        self.debug_dir = debug_dir
        self.run_id = run_id or self._generate_run_id()
        self.logger = logging.getLogger(__name__)
        self._ensure_debug_directory()

    def _generate_run_id(self) -> str:
        """Generate a unique run ID for this execution"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"run_{timestamp}"

    def _ensure_debug_directory(self):
        """Ensure debug directory exists with improved structure"""
        # Create base debug directory
        os.makedirs(self.debug_dir, exist_ok=True)

        # Create seed-specific directory
        seed_dir = os.path.join(self.debug_dir, self.trajectory_id)
        os.makedirs(seed_dir, exist_ok=True)

        # Create run-specific directory for this execution
        run_dir = os.path.join(seed_dir, self.run_id)
        os.makedirs(run_dir, exist_ok=True)

        # Create subdirectories for different types of data
        self.phases_dir = os.path.join(run_dir, "phases")
        self.visualizations_dir = os.path.join(run_dir, "visualizations")
        self.window_states_dir = os.path.join(run_dir, "window_states")
        self.summaries_dir = os.path.join(run_dir, "summaries")

        os.makedirs(self.phases_dir, exist_ok=True)
        os.makedirs(self.visualizations_dir, exist_ok=True)
        os.makedirs(self.window_states_dir, exist_ok=True)
        os.makedirs(self.summaries_dir, exist_ok=True)

    def save_phase_data(self, step_idx: int, phase: str, data: Dict[str, Any]) -> str:
        """Save phase data to file with improved organization"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"step_{step_idx:03d}_{phase}_{timestamp}.json"
        filepath = os.path.join(self.phases_dir, filename)

        # Convert non-serializable objects
        serializable_data = self._make_serializable(data)

        with open(filepath, "w") as f:
            json.dump(serializable_data, f, indent=2)

        self.logger.debug(f"Saved {phase} data: {filepath}")
        return filepath

    def load_phase_data(self, step_idx: int, phase: str) -> Optional[Dict[str, Any]]:
        """Load most recent phase data for step"""
        pattern = f"step_{step_idx:03d}_{phase}_"

        if not os.path.exists(self.phases_dir):
            return None

        # Find most recent file matching pattern
        files = [f for f in os.listdir(self.phases_dir) if f.startswith(pattern) and f.endswith(".json")]
        if not files:
            return None

        # Sort by timestamp (newest first)
        files.sort(reverse=True)
        filepath = os.path.join(self.phases_dir, files[0])

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
            "text": element.name,  # UIElement only has 'name' field, not 'text'
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
            "window_states": context.window_states,
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

    def save_perturbation_command(self, step_idx: int, command_data: Dict[str, Any]) -> str:
        """Save concrete perturbation command execution details"""
        return self.save_phase_data(step_idx, "perturbation_command", command_data)

    def save_perturbation_summary(self, step_idx: int, summary_data: Dict[str, Any]) -> str:
        """Save perturbation summary for a step"""
        return self.save_phase_data(step_idx, "perturbation_summary", summary_data)

    def save_window_states(self, step_idx: int, phase: str, window_states: List[Any]) -> str:
        """Save window states for specific phase in dedicated directory"""
        # Convert WindowState objects to serializable format using recursive method
        serializable_window_states = [self._recursive_to_dict(ws) for ws in window_states]

        data = {"window_states": serializable_window_states, "phase": phase}

        # Save in dedicated window_states directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"step_{step_idx:03d}_{phase}_{timestamp}.json"
        filepath = os.path.join(self.window_states_dir, filename)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        self.logger.debug(f"Saved window states ({phase}): {filepath}")
        return filepath

    def save_element_test_results(self, step_idx: int, test_results: List[Dict[str, Any]]) -> str:
        """Save element test results for debugging"""
        data = {"test_results": test_results}
        return self.save_phase_data(step_idx, "element_test_results", data)

    def load_window_states(self, step_idx: int, phase: str) -> Optional[List[Dict[str, Any]]]:
        """Load window states for specific phase from dedicated directory"""
        pattern = f"step_{step_idx:03d}_{phase}_"

        if not os.path.exists(self.window_states_dir):
            return None

        # Find most recent file matching pattern
        files = [
            f for f in os.listdir(self.window_states_dir) if f.startswith(pattern) and f.endswith(".json")
        ]
        if not files:
            return None

        # Sort by timestamp (newest first)
        files.sort(reverse=True)
        filepath = os.path.join(self.window_states_dir, files[0])

        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            return data.get("window_states")
        except Exception as e:
            self.logger.error(f"Error loading window states ({phase}): {e}")
            return None

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

    def visualize_element_bounding_boxes(
        self,
        window_states: List[Any],
        target_element_id: str = None,
        screenshot_data: bytes = None,
        step_idx: int = None,
    ) -> str:
        """
        Visualize bounding boxes of extracted elements on screenshot for debugging.
        Saves the visualization in the dedicated visualizations directory.

        Args:
            window_states: List of WindowState objects
            target_element_id: Specific element ID to highlight (optional)
            screenshot_data: Screenshot data bytes (optional)
            step_idx: Step index for filename (optional)

        Returns:
            Path to the annotated screenshot
        """
        try:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d@%H%M%S")
            if step_idx is not None:
                filename = f"element_visualization_step_{step_idx:03d}_{timestamp}.png"
            else:
                filename = f"element_visualization_{timestamp}.png"

            output_path = os.path.join(self.visualizations_dir, filename)

            # Use provided screenshot data or create a placeholder
            if screenshot_data:
                screenshot = Image.open(BytesIO(screenshot_data))
            else:
                # Create a placeholder image if no screenshot provided
                screenshot = Image.new("RGB", (1920, 1080), color="lightgray")
                self.logger.warning("No screenshot data provided, using placeholder image")

            draw = ImageDraw.Draw(screenshot)

            font = ImageFont.load_default()
            small_font = ImageFont.load_default()

            colors = [
                (255, 0, 0),  # Red
                (0, 255, 0),  # Green
                (0, 0, 255),  # Blue
                (255, 255, 0),  # Yellow
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Cyan
            ]

            element_count = 0
            highlighted_element = None

            # Draw bounding boxes for all elements
            for window_state in window_states:
                elements = window_state.get_all_elements(include_structural=False)
                color = colors[element_count % len(colors)]

                for element in elements:
                    pos = element.position
                    if not pos:
                        continue

                    center_x = pos.get("center_x", 0)
                    center_y = pos.get("center_y", 0)
                    width = pos.get("width", 0)
                    height = pos.get("height", 0)

                    # Calculate bounding box coordinates
                    left = center_x - width // 2
                    top = center_y - height // 2
                    right = center_x + width // 2
                    bottom = center_y + height // 2

                    # Check if this is the target element
                    is_target = (
                        target_element_id and element.element_id and element.element_id == target_element_id
                    )

                    if is_target:
                        highlighted_element = element
                        # Use bright red for target element
                        box_color = (255, 0, 0)
                        text_color = (255, 255, 255)
                        thickness = 3
                    else:
                        box_color = color
                        text_color = (255, 255, 255)
                        thickness = 1

                    # Draw bounding box
                    draw.rectangle([left, top, right, bottom], outline=box_color, width=thickness)

                    # Draw element label with coordinates
                    label = element.name or element.element_type or "Unknown"
                    if element.element_id:
                        label = f"{str(element.element_id)[:4]}: {label}"

                    # Add coordinates to label
                    coord_label = f"({center_x}, {center_y})"

                    # Draw text background for main label
                    text_bbox = draw.textbbox((left, top - 35), label, font=font)
                    draw.rectangle(text_bbox, fill=box_color)

                    # Draw main label
                    draw.text((left, top - 35), label, fill=text_color, font=font)

                    # Draw coordinates label
                    coord_bbox = draw.textbbox((left, top - 20), coord_label, font=small_font)
                    draw.rectangle(coord_bbox, fill=(0, 0, 0, 180))
                    draw.text((left, top - 20), coord_label, fill=(255, 255, 255), font=small_font)

                    element_count += 1

            # Add summary information
            summary_text = f"Total elements: {element_count}"
            if highlighted_element:
                summary_text += (
                    f"\nTarget: {highlighted_element.name or 'Unknown'} ({highlighted_element.element_id})"
                )
                pos = highlighted_element.position
                summary_text += (
                    f"\nCoords: ({pos['center_x']}, {pos['center_y']}) size {pos['width']}x{pos['height']}"
                )

            # Draw summary box
            draw.rectangle([10, 10, 350, 100], fill=(0, 0, 0, 180))
            draw.text((15, 15), summary_text, fill=(255, 255, 255), font=font)

            # Save annotated screenshot
            screenshot.save(output_path)
            self.logger.info(f"Element visualization saved to: {output_path}")

            return output_path

        except Exception as e:
            self.logger.error(f"Error creating element visualization: {e}")
            return None

    def get_phase_summary(self, step_idx: int) -> Dict[str, Any]:
        """Get summary of all phases for a step"""
        if not os.path.exists(self.phases_dir):
            return {}

        pattern = f"step_{step_idx:03d}_"
        files = [f for f in os.listdir(self.phases_dir) if f.startswith(pattern) and f.endswith(".json")]

        phases = {}
        for file in files:
            phase_name = file.replace(f"step_{step_idx:03d}_", "").replace(".json", "")
            timestamp = phase_name.split("_")[-1] if "_" in phase_name else "unknown"
            phase_name = "_".join(phase_name.split("_")[:-1]) if "_" in phase_name else phase_name

            if phase_name not in phases:
                phases[phase_name] = []
            phases[phase_name].append(
                {"file": file, "timestamp": timestamp, "path": os.path.join(self.phases_dir, file)}
            )

        return phases

    def save_run_summary(self, summary_data: Dict[str, Any]) -> str:
        """Save a summary of the entire run"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"run_summary_{timestamp}.json"
        filepath = os.path.join(self.summaries_dir, filename)

        with open(filepath, "w") as f:
            json.dump(summary_data, f, indent=2)

        self.logger.info(f"Saved run summary: {filepath}")
        return filepath

    def get_run_info(self) -> Dict[str, str]:
        """Get information about this run"""
        return {
            "trajectory_id": self.trajectory_id,
            "run_id": self.run_id,
            "phases_dir": self.phases_dir,
            "visualizations_dir": self.visualizations_dir,
            "window_states_dir": self.window_states_dir,
            "summaries_dir": self.summaries_dir,
        }

    def _make_serializable(self, data: Any) -> Any:
        """Convert data to JSON-serializable format"""
        if isinstance(data, dict):
            return {k: self._make_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._make_serializable(item) for item in data]
        elif isinstance(data, (str, int, float, bool, type(None))):
            return data
        elif hasattr(data, "window_id") or hasattr(data, "element_id"):
            # Handle WindowState/UIElement objects using recursive method
            return self._recursive_to_dict(data)
        else:
            # Convert other types to string representation
            return str(data)

    def _window_state_to_dict(self, window_state: Any) -> Dict[str, Any]:
        """Convert WindowState object to serializable dictionary"""
        return self._recursive_to_dict(window_state)

    def _ui_element_to_dict(self, element: Any) -> Dict[str, Any]:
        """Convert UIElement object to serializable dictionary"""
        return self._recursive_to_dict(element)

    def _recursive_to_dict(self, obj: Any) -> Dict[str, Any]:
        """Recursively convert WindowState/UIElement objects to serializable dictionaries"""
        if obj is None:
            return None

        # Handle WindowState objects
        if hasattr(obj, "window_id"):
            return {
                "window_id": obj.window_id,
                "window_name": obj.window_name,
                "app_name": obj.app_name,
                "is_active": obj.is_active,
                "is_modal": obj.is_modal,
                "is_minimized": obj.is_minimized,
                "geometry": obj.geometry,
                "z_order": obj.z_order,
                "x11_window_id": obj.x11_window_id,
                "is_mapped": obj.is_mapped,
                "desktop": obj.desktop,
                "root_element": self._recursive_to_dict(obj.root_element),
            }

        # Handle UIElement objects
        elif hasattr(obj, "element_id"):
            return {
                "element_id": obj.element_id,
                "element_type": obj.element_type,
                "name": obj.name,
                "position": obj.position,
                "parent_id": obj.parent_id,
                "children": [self._recursive_to_dict(child) for child in obj.children],
                "depth": obj.depth,
                "visibility": obj.visibility.value
                if hasattr(obj.visibility, "value")
                else str(obj.visibility),
                "is_enabled": obj.is_enabled,
                "is_focused": obj.is_focused,
                "is_expanded": obj.is_expanded,
                "properties": obj.properties,
            }

        # Handle other objects (fallback)
        else:
            return str(obj)

    def load_window_states_as_objects(self, step_idx: int, phase: str) -> Optional[List[Dict[str, Any]]]:
        """Load window states and return as reconstructed objects (for debugging)"""
        data = self.load_phase_data(step_idx, f"window_states_{phase}")
        if not data:
            return None

        window_states_data = data.get("window_states", [])
        reconstructed_states = []

        for ws_data in window_states_data:
            reconstructed_state = self._dict_to_window_state(ws_data)
            reconstructed_states.append(reconstructed_state)

        return reconstructed_states

    def _dict_to_window_state(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert dictionary back to WindowState-like structure (for debugging)"""
        if not data:
            return None

        # Reconstruct WindowState
        if "window_id" in data:
            return {
                "window_id": data["window_id"],
                "window_name": data["window_name"],
                "app_name": data["app_name"],
                "is_active": data["is_active"],
                "is_modal": data["is_modal"],
                "is_minimized": data["is_minimized"],
                "geometry": data["geometry"],
                "z_order": data["z_order"],
                "x11_window_id": data["x11_window_id"],
                "is_mapped": data["is_mapped"],
                "desktop": data["desktop"],
                "root_element": self._dict_to_ui_element(data["root_element"]),
            }

        # Reconstruct UIElement
        elif "element_id" in data:
            return {
                "element_id": data["element_id"],
                "element_type": data["element_type"],
                "name": data["name"],
                "position": data["position"],
                "parent_id": data["parent_id"],
                "children": [self._dict_to_ui_element(child) for child in data.get("children", [])],
                "depth": data["depth"],
                "visibility": data["visibility"],
                "is_enabled": data["is_enabled"],
                "is_focused": data["is_focused"],
                "is_expanded": data["is_expanded"],
                "properties": data["properties"],
            }

        return data

    def _dict_to_ui_element(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert dictionary back to UIElement-like structure (for debugging)"""
        return self._dict_to_window_state(data)  # Same logic for both


class PhaseDebugger:
    """Debug individual phases using saved data"""

    def __init__(self, trajectory_id: str, debug_dir: str = "./debug", run_id: str = None):
        self.data_manager = PhaseDataManager(trajectory_id, debug_dir, run_id)
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
