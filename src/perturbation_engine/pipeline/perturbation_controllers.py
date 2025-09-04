from abc import ABC, abstractmethod
from typing import Any, Dict

from perturbation_engine.data_types import PerturbationSpec, PerturbationType
from perturbation_engine.pipeline.perturbation_desktop_env import DesktopEnv


class PerturbationController(ABC):
    """Abstract base class for perturbation controllers"""

    @abstractmethod
    def can_handle(self, perturbation_type: PerturbationType) -> bool:
        """Check if this controller can handle the perturbation type"""
        pass

    @abstractmethod
    def apply_perturbation(
        self, env: DesktopEnv, spec: PerturbationSpec, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply the perturbation and return execution details"""
        pass

    @abstractmethod
    def validate_perturbation(self, env: DesktopEnv, spec: PerturbationSpec, context: Dict[str, Any]) -> bool:
        """Validate that perturbation was applied correctly"""
        pass


class VLMController(PerturbationController):
    """Controller for instruction-level perturbations"""

    def __init__(self, vlm):
        self.vlm = vlm

    def can_handle(self, perturbation_type: PerturbationType) -> bool:
        return perturbation_type == PerturbationType.INSTRUCTION

    def apply_perturbation(
        self, env: DesktopEnv, spec: PerturbationSpec, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        # TODO: Implement instruction perturbation logic
        # - Use sampling engines for task description randomization
        # - Modify instruction text, add noise, change wording
        return {"applied": True, "method": "instruction_sampling"}

    def validate_perturbation(self, env: DesktopEnv, spec: PerturbationSpec, context: Dict[str, Any]) -> bool:
        # TODO: Validate instruction changes don't break task semantics
        return True
