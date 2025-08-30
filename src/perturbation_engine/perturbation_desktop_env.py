import logging
from typing import Any, Dict, Optional, Union

from OSWorld.desktop_env.desktop_env import DesktopEnv

# from perturbation_engine.sampling.ui_visual_design.web_ui_sampler import WebUISampler
from perturbation_engine.perturbation_controller import PerturbationController
from perturbation_engine.scenario_manager import ScenarioManager
from perturbation_engine.types import ScenarioParameters


class PerturbationDesktopEnv(DesktopEnv):
    """DesktopEnv for perturbation"""

    def __init__(
        self,
        provider_name: str = "vmware",
        region: str = None,
        path_to_vm: str = None,
        snapshot_name: str = "init_state",
        action_space: str = "pyautogui",
        cache_dir: str = "cache",
        screen_size: tuple = (1920, 1080),
        headless: bool = False,
        require_a11y_tree: bool = True,
        require_terminal: bool = False,
        os_type: str = "Ubuntu",
        enable_proxy: bool = False,
        client_password: str = "",
        perturbation_config: Optional[Union[str, Dict[str, Any]]] = None,
        perturbation_seed: Optional[int] = None,
    ):
        """Initialize enhanced environment with perturbation capabilities."""
        super().__init__(
            provider_name=provider_name,
            region=region,
            path_to_vm=path_to_vm,
            snapshot_name=snapshot_name,
            action_space=action_space,
            cache_dir=cache_dir,
            screen_size=screen_size,
            headless=headless,
            require_a11y_tree=require_a11y_tree,
            require_terminal=require_terminal,
            os_type=os_type,
            enable_proxy=enable_proxy,
            client_password=client_password,
        )

        self._init_perturbation_engine(self, perturbation_config, perturbation_seed)

    def _init_perturbation_engine(
        self,
        base_env,
        task_config: Optional[Dict[str, Any]] = None,
        perturbation_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize enhanced environment.

        Args:
            base_env: The base OSWorld DesktopEnv instance
            perturbation_config: Configuration for perturbation system
        """
        # TODO: Load configuration from yaml file
        self.base_env = base_env
        self.task_config = task_config
        self.perturbation_enabled = perturbation_config is not None

        if self.perturbation_enabled:
            self._logger = logging.getLogger(__name__)
            self._logger.info("Initializing perturbation system")

            self.scenario_manager = ScenarioManager(task_config, perturbation_config)
            self.samplers = {
                # "web_ui": WebUISampler()
            }
            self.perturbation_controller = PerturbationController(self.scenario_manager, self.samplers)

    def reset(self, task_config: Optional[Dict[str, Any]] = None, seed=None, options=None) -> Dict[str, Any]:
        """Reset environment with optional perturbation injection."""
        observation = super().reset(task_config, seed, options)

        if task_config is not None and self.perturbation_enabled:
            # Initialize scenario manager with current task config
            if self.scenario_manager is None:
                self.scenario_manager = ScenarioManager(task_config, self.perturbation_config)

            # Update scenario manager with new task config
            self.scenario_manager.update_task_config(task_config)

            # Generate perturbation scenario
            scenario = self.scenario_manager.generate_scenario()

            # Apply perturbations using controller
            self._apply_perturbations(scenario)

        return observation

    def step(self, action, pause=2):
        """Execute action with potential runtime perturbation injection."""
        # Execute the action first
        observation, reward, done, info = super().step(action, pause)

        # Apply runtime perturbations if enabled
        if self.perturbation_enabled and self._should_apply_runtime_perturbation():
            runtime_scenario = self.scenario_manager.generate_runtime_scenario()
            self._apply_runtime_perturbations(runtime_scenario)
            # Get updated observation after runtime perturbation
            observation = self._get_obs()

        return observation, reward, done, info

    def _apply_perturbations(self, scenario: ScenarioParameters):
        """Apply setup perturbations based on generated scenario."""
        if self.perturbation_controller is None:
            self.perturbation_controller = PerturbationController(scenario)

        # Generate and execute commands
        commands = self.perturbation_controller.generate_commands(scenario)
        results = self.perturbation_controller.execute_batch(commands)

        # Store results for status/history
        self._perturbation_results = results

    def _apply_runtime_perturbations(self, scenario: ScenarioParameters):
        """Apply runtime perturbations during task execution."""
        if self.perturbation_controller is None:
            self.perturbation_controller = PerturbationController(scenario)

        # Generate and execute runtime commands
        commands = self.perturbation_controller.generate_commands(scenario)
        results = self.perturbation_controller.execute_batch(commands)

        # Store runtime results
        if not hasattr(self, "_runtime_perturbation_results"):
            self._runtime_perturbation_results = []
        self._runtime_perturbation_results.extend(results)

    def _should_apply_runtime_perturbation(self) -> bool:
        """Determine if runtime perturbation should be applied."""
        # TODO: Implement runtime perturbation triggers
        # - Time-based triggers
        # - Action-based triggers
        # - State-based triggers
        return False

    def get_perturbation_status(self) -> Dict[str, Any]:
        """Get current perturbation system status."""
        if not self.perturbation_enabled:
            return {"enabled": False}

        return {
            "enabled": True,
            "setup_perturbations": getattr(self, "_perturbation_results", []),
            "runtime_perturbations": getattr(self, "_runtime_perturbation_results", []),
            "scenario_manager": self.scenario_manager is not None,
            "perturbation_controller": self.perturbation_controller is not None,
        }

    def get_perturbation_history(self) -> list:
        """Get detailed perturbation execution history."""
        if not self.perturbation_enabled:
            return []

        history = []
        if hasattr(self, "_perturbation_results"):
            history.extend(self._perturbation_results)
        if hasattr(self, "_runtime_perturbation_results"):
            history.extend(self._runtime_perturbation_results)

        return history
