"""
Factory for creating perturbation scenarios.
"""

import logging
from typing import Dict, Type

from perturbation_engine.scenarios.base_scenario import PerturbationScenario
from perturbation_engine.scenarios.scenarios import ChromePerturbationScenario


class PerturbationScenarioFactory:
    """Factory for creating perturbation scenario instances."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._scenarios: Dict[str, Type[PerturbationScenario]] = {
            "chrome": ChromePerturbationScenario,
        }

    def register_scenario(self, name: str, scenario_class: Type[PerturbationScenario]) -> None:
        """Register a scenario class.

        Args:
            name: Scenario name
            scenario_class: Scenario class
        """
        if name not in self._scenarios:
            self._scenarios[name] = scenario_class
        else:
            self.logger.warning(f"Scenario {name} already registered, skipping registration")
        self.logger.debug(f"Registered scenario: {name}")

    def create_scenario(self, name: str) -> PerturbationScenario:
        """Create a scenario instance.

        Args:
            name: Scenario name

        Returns:
            Scenario instance

        Raises:
            ValueError: If scenario not found
        """
        if name not in self._scenarios:
            raise ValueError(f"Unknown scenario: {name}")

        return self._scenarios[name]()

    def get_available_scenarios(self) -> list[str]:
        """Get list of available scenario names."""
        return list(self._scenarios.keys())
