"""
Base perturbation scenario interface.

Defines the contract for all perturbation scenarios.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class PerturbationScenario(ABC):
    """Abstract base class for perturbation scenarios."""

    @abstractmethod
    def apply_setup_perturbations(
        self,
        task_config: Dict[str, Any],
        perturbation_scenario: "PerturbationScenario",
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply setup perturbations to the environment.

        Args:
            env: The environment to perturb
            parameters: Scenario-specific parameters

        Returns:
            Modified task configuration
        """
        return task_config

    @abstractmethod
    def check_and_apply_runtime_perturbations(self, env: Any, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Check and apply runtime perturbations for this scenario.

        Args:
            parameters: Scenario-specific parameters

        Returns:
            Dictionary containing the perturbation result
        """
        return {}

    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate scenario parameters.

        Args:
            parameters: Parameters to validate

        Returns:
            True if parameters are valid
        """
        pass
