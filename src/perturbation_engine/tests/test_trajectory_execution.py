"""Test trajectory generator with real DesktopEnv and file paths"""

import json
import logging
import os
from multiprocessing import Manager

from perturbation_engine.data_types import (
    ExecutionConfig,
    PerturbationPhase,
    PerturbationSpec,
    PerturbationType,
    ScenarioSpec,
)
from perturbation_engine.pipeline.parallel_execution_engine import ParallelExecutionEngine

# Configure logging for testing
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_real_trajectory_execution():
    """Test trajectory execution with real DesktopEnv and file paths"""

    # Load real OSWorld task configuration
    task_config_path = (
        "src/OSWorld/evaluation_examples/examples/chrome/0d8b7de3-e8de-4d86-b9fd-dd2dce58a217.json"
    )
    with open(task_config_path, "r") as f:
        task_config = json.load(f)

    # Real trajectory file path
    trajectory_path = (
        "external_data/osworld-verified/jedi-7b-4o-15steps/chrome/0d8b7de3-e8de-4d86-b9fd-dd2dce58a217"
    )

    # Verify files exist
    assert os.path.exists(task_config_path), f"Task config not found: {task_config_path}"
    assert os.path.exists(trajectory_path), f"Trajectory path not found: {trajectory_path}"
    assert os.path.exists(os.path.join(trajectory_path, "traj.jsonl")), "Trajectory file not found"

    # Create execution configuration for real DesktopEnv
    execution_config = ExecutionConfig(
        # VM/Provider settings - dynamically chosen
        path_to_vm="/Users/lockewang/FIG/OSWorld/vmware_vm_data/Ubuntu0/Ubuntu0.vmx",
        provider_name="vmware",
        region="us-east-1",
        snapshot_name="chrome",
        # Environment settings
        headless=True,
        action_space="pyautogui",
        observation_type="screenshot",
        screen_size=(1920, 1080),
        os_type="Ubuntu",
        client_password="",
        # Execution settings
        max_steps=6,  # Limit to 6 steps based on trajectory
        sleep_after_execution=0.0,
        # Additional settings
        cache_dir="cache",
        require_a11y_tree=False,  # Disable for faster testing
        require_terminal=False,
        enable_proxy=False,
    )

    # Create realistic perturbations
    perturbations = [
        PerturbationSpec(
            perturbation_type=PerturbationType.UI_VISUAL,
            phase=PerturbationPhase.RUNTIME,
            perturbation_controller="gemini",
            parameters={"action": "ui_injection", "num_components": 3},
            trigger_function_name="step_range",
            trigger_parameters={"start": 2, "end": 4},
            validation_function_name="element_created",
            validation_parameters={"selector": ".injected-element"},
            name="ui_injection_test",
            description="Inject UI elements during trajectory execution",
        )
    ]

    # Create scenario specification
    scenario_id = "test_scenario_001"
    scenario_spec = ScenarioSpec(
        task_id="0d8b7de3-e8de-4d86-b9fd-dd2dce58a217",
        scenario_id=scenario_id,
        task_config=task_config,
        trajectory_file_path=trajectory_path,
        perturbations=perturbations,
        result_dir=f"./test_results/0d8b7de3-e8de-4d86-b9fd-dd2dce58a217/{scenario_id}",
        metadata={
            "scenario_id": scenario_id,
            "test_type": "real_execution",
            "source": "test_trajectory_generator",
        },
    )

    # Create multiprocessing components
    with Manager() as manager:
        scenario_queue = manager.Queue()
        scenario_queue.put(scenario_spec)
        shared_results = manager.list()

        # Create and run execution engine
        engine = ParallelExecutionEngine(execution_config)

        logger.info("Starting trajectory execution test...")
        logger.info(f"Task: {task_config['instruction']}")
        logger.info(f"Trajectory path: {trajectory_path}")
        logger.info(f"Result dir: {scenario_spec.result_dir}")

        try:
            # Run the execution
            engine.run_vm_tasks(scenario_queue, shared_results)

            # Check results
            results = list(shared_results)
            assert len(results) > 0, "No results generated"

            result = results[0]
            logger.info("Execution completed:")
            logger.info(f"  Task ID: {result.task_id}")
            logger.info(f"  Success: {result.success}")
            logger.info(f"  Score: {result.result_score}")
            logger.info(f"  Generation time: {result.generation_time:.2f}s")
            logger.info(f"  Perturbations applied: {len(result.perturbation_log)}")

            # Verify result structure
            assert hasattr(result, "task_id")
            assert hasattr(result, "success")
            assert hasattr(result, "result_score")
            assert hasattr(result, "perturbation_log")
            assert hasattr(result, "generation_time")
            assert hasattr(result, "metadata")

            # Verify result directory was created
            assert os.path.exists(scenario_spec.result_dir), "Result directory not created"

            # Check for trajectory files
            traj_file = os.path.join(scenario_spec.result_dir, "traj.jsonl")
            result_file = os.path.join(scenario_spec.result_dir, "result.txt")
            perturbations_file = os.path.join(scenario_spec.result_dir, "perturbations.json")

            if os.path.exists(traj_file):
                logger.info(f"Trajectory file created: {traj_file}")
            if os.path.exists(result_file):
                logger.info(f"Result file created: {result_file}")
            if os.path.exists(perturbations_file):
                logger.info(f"Perturbations file created: {perturbations_file}")

            logger.info("Test completed successfully!")

        except Exception as e:
            logger.error(f"Test failed with error: {e}")
            import traceback

            logger.error(traceback.format_exc())
            raise


if __name__ == "__main__":
    test_real_trajectory_execution()
