from typing import List

from OSWorld.desktop_env.controllers.python import PythonController
from OSWorld.desktop_env.controllers.setup import SetupController
from perturbation_engine.types import Command, ExecutionResult, ScenarioParameters


class PerturbationController:
    """Controller for perturbation.

    Given a scenario, generate and execute commands using corresponding controllers

    Args:
        scenario: scenario parameters

    methods:
        - generate_scenario: generates commands based on scenario parameters
        - execute_commands: calls relevant controllers to execute commands
    """

    def __init__(self, scenario: ScenarioParameters):
        self.scenario = scenario
        self.controllers = {
            "setup": SetupController(),
            "python": PythonController(),
        }

    def generate_commands(self, scenario: ScenarioParameters) -> List[Command]:
        """Generate execution commands based on scenario parameters."""
        # TODO: Convert scenario parameters to executable commands
        # TODO: Map perturbation types to command types
        pass

    def execute_batch(self, commands: List[Command]) -> List[ExecutionResult]:
        """Execute a batch of perturbation commands."""
        # TODO: Execute commands in parallel or sequentially
        pass

    def execute_single(self, command: Command) -> ExecutionResult:
        """Execute a single perturbation command."""
        # TODO: Select appropriate controller and execute
        pass

    def _select_controller(self, command: Command):
        """Select appropriate controller for command execution."""
        # TODO: Map command types to controller types
        pass
