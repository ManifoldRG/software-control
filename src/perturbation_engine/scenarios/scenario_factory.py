"""
Factory pattern for creating perturbation scenarios.
"""

import logging
from typing import Dict, List, Type

from perturbation_engine.data_types import DifficultyLevel
from perturbation_engine.scenarios.base_scenarios import BasePerturbationScenario


class ScenarioFactory:
    """Simplified factory for creating perturbation scenarios."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._registry: Dict[tuple[str, str], Type[BasePerturbationScenario]] = {}
        self._scenario_types = ["invariance", "distractor", "negative"]
        self._scenario_cache = {}  # Cache instances for efficiency

    def register_scenario(
        self, task_type: str, scenario_type: str, scenario_class: Type[BasePerturbationScenario]
    ):
        """Register a scenario class for a task type and scenario type combination."""
        self._registry[(task_type, scenario_type)] = scenario_class

    def create_scenario(self, task_type: str, scenario_type: str) -> BasePerturbationScenario:
        """Create a scenario instance for the given task type and scenario type."""
        cache_key = (task_type, scenario_type)

        if cache_key not in self._scenario_cache:
            scenario_class = self._registry.get(cache_key)
            if not scenario_class:
                # Fallback to Chrome scenario
                scenario_class = self._registry.get(("chrome", scenario_type))
                if not scenario_class:
                    raise ValueError(f"No scenario class found for ({task_type}, {scenario_type})")
                self.logger.warning(f"Using fallback scenario {scenario_class.__name__} for {cache_key}")

            self._scenario_cache[cache_key] = scenario_class()

        return self._scenario_cache[cache_key]

    def get_difficulty_levels(self, task_type: str, scenario_type: str) -> List[DifficultyLevel]:
        """Get difficulty levels for a task type and scenario type combination."""
        scenario = self.create_scenario(task_type, scenario_type)
        return scenario.get_difficulty_levels()


def create_default_factory() -> ScenarioFactory:
    """Create a factory with default scenario registrations."""
    from perturbation_engine.scenarios.chrome_scenarios import (
        ChromeDistractorScenario,
        ChromeInvarianceScenario,
        ChromeNegativeScenario,
    )
    from perturbation_engine.scenarios.os_scenarios import (
        OSInvarianceScenario,
        # OSDistractorScenario,
        # OSNegativeScenario,
    )

    factory = ScenarioFactory()

    # Register Chrome scenarios
    factory.register_scenario("chrome", "invariance", ChromeInvarianceScenario)
    factory.register_scenario("chrome", "distractor", ChromeDistractorScenario)
    factory.register_scenario("chrome", "negative", ChromeNegativeScenario)

    # Register OS scenarios
    factory.register_scenario("os", "invariance", OSInvarianceScenario)
    factory.register_scenario("os", "distractor", ChromeDistractorScenario)  # Fallback
    factory.register_scenario("os", "negative", ChromeNegativeScenario)  # Fallback

    # Register other task types with Chrome scenarios as fallbacks
    other_task_types = [
        "gimp",
        "thunderbird",
        "vlc",
        "vs_code",
        "libreoffice_calc",
        "libreoffice_impress",
        "libreoffice_writer",
    ]

    for task_type in other_task_types:
        factory.register_scenario(task_type, "invariance", ChromeInvarianceScenario)
        factory.register_scenario(task_type, "distractor", ChromeDistractorScenario)
        factory.register_scenario(task_type, "negative", ChromeNegativeScenario)

    return factory
