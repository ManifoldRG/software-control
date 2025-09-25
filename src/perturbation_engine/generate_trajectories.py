# Portions of this code are adapted from the OSWorld repository
# https://github.com/xlang-ai/OSWorld
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import signal
import sys

from dotenv import load_dotenv

from perturbation_engine.configure_logging import configure_logging
from perturbation_engine.curriculum.curriculum_orchestrator import create_custom_curriculum
from perturbation_engine.data_types import ExecutionConfig, GenerationConfig
from perturbation_engine.unified_orchestrator import UnifiedOrchestrator

load_dotenv()
configure_logging()
logger = logging.getLogger(__name__)

active_environments = []
processes = []
is_terminating = False

# ============================================================================
# Signal Handler
# ============================================================================


def signal_handler(signum, frame):
    """Handle termination signals for graceful shutdown"""
    global is_terminating, active_environments, processes

    if is_terminating:
        return

    is_terminating = True
    logger.info(f"Received signal {signum}. Gracefully shutting down...")

    # Close environments and terminate processes
    for env in active_environments:
        try:
            env.close()
        except Exception as e:
            logger.error(f"Error closing environment: {e}")

    for p in processes:
        if p.is_alive():
            try:
                p.terminate()
            except Exception as e:
                logger.error(f"Error terminating process: {e}")

    sys.exit(0)


def main():
    """Main entry point for trajectory generation"""
    os.environ["PROXY_CONFIG_FILE"] = "src/OSWorld/evaluation_examples/settings/proxy/dataimpulse.json"

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Example usage
    execution_config = ExecutionConfig(
        # VM/Provider settings
        path_to_vm="/Users/lockewang/FIG/OSWorld/vmware_vm_data/Ubuntu0/Ubuntu0.vmx",
        # path_to_vm=None,
        # provider_name="aws",
        provider_name="vmware",
        region=os.environ["AWS_REGION"],
        # snapshot_name=os.environ["AWS_SNAPSHOT_NAME"],
        snapshot_name="chrome",
        # Environment settings
        headless=True,
        action_space="pyautogui",
        observation_type="screenshot",
        screen_size=(1920, 1080),
        os_type="Ubuntu",
        # client_password="osworld-public-evaluation",
        client_password="",
        # Execution settings
        max_steps=15,
        sleep_after_execution=0.0,
        # Additional OSWorld settings
        cache_dir="cache",
        require_a11y_tree=True,
        require_terminal=False,
        enable_proxy=False,
        # Perturbation connection
        chromium_port=9222,
    )

    # Initialize unified orchestrator
    orchestrator = UnifiedOrchestrator(execution_config)

    # Test configuration
    task_config_base_dir = "src/OSWorld/evaluation_examples"
    trajectory_base_dir = "external_data/osworld-verified/jedi-7b-4o-15steps/jedi-7b-4o-15steps"
    result_base_dir = "/opt/manifold/results"

    # Load seed trajectories using the unified orchestrator
    seed_trajectories = orchestrator.load_seed_trajectories(task_config_base_dir, trajectory_base_dir)
    seed_trajectories = seed_trajectories[:2]  # Limit for testing

    # Choose generation method
    use_curriculum = True  # Set to True for curriculum-based generation

    if use_curriculum:
        curriculum_config = create_custom_curriculum(
            num_trajectories=10,  # Generate 10 curriculum trajectories
            easy_ratio=0.4,
            medium_ratio=0.4,
            hard_ratio=0.2,
        )

        # Use first seed trajectory for curriculum generation
        results = orchestrator.generate_curriculum_trajectories(
            seed_trajectory=seed_trajectories[0],
            curriculum_config=curriculum_config,
            num_parallel_vms=1,
            result_base_dir=result_base_dir,
        )
    else:
        # Generate static trajectories
        results = orchestrator.generate_static_trajectories(
            seed_trajectories=seed_trajectories,
            generation_config=GenerationConfig(
                num_invariance_scenarios=1,
                num_distractor_scenarios=1,
                num_negative_scenarios=0,
                num_difficulty_levels=1,
            ),
            num_parallel_vms=1,
            result_base_dir=result_base_dir,
        )

    return results


if __name__ == "__main__":
    main()
