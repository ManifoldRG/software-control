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

import json
import logging
import os
import signal
import sys
import time
from multiprocessing import Manager, Process
from dataclasses import asdict
import random
from typing import Dict, List, Set, Tuple

from dotenv import load_dotenv

from perturbation_engine.configure_logging import configure_logging
from perturbation_engine.data_types import ExecutionConfig, GenerationConfig, GenerationResult, SeedTrajectory
from perturbation_engine.pipeline.parallel_execution_engine import ParallelExecutionEngine
from perturbation_engine.pipeline.scenario_generator import ScenarioGenerator

load_dotenv()
configure_logging()
logger = logging.getLogger(__name__)

active_environments = []
processes = []
is_terminating = False

# ---------------------------
# Defaults / Constants
# ---------------------------

# Paths to precomputed failure lists (newline-delimited JSON)
DEFAULT_O3_FAILURES_PATH = "external_data/failures/o3_gta1_100steps_failed_tasks.json"
DEFAULT_UITARS_FAILURES_PATH = "external_data/failures/uitars15-7b-100step-1_failed_tasks.json"

# Reproducible sampling seed (can be overridden via env var SEED_TRAJECTORIES_SEED)
DEFAULT_SEED_TRAJECTORIES_SEED = 42

# Per-task-type sample caps (mirrors experiments/seed_data_selection.py)
DEFAULT_SAMPLES_REQUIRED: Dict[str, int] = {
    "chrome": 9,
    "vs_code": 3,
    "vlc": 4,
    "gimp": 3,
    "libreoffice_calc": 14,
    "libreoffice_impress": 12,
    "libreoffice_writer": 5,
    "os": 4,
    "thunderbird": 1,
    "multi_apps": 44,
}


class TrajectoryGenerationOrchestrator:
    """Main orchestrator for trajectory generation with perturbations"""

    def __init__(self, scenario_generator: ScenarioGenerator) -> None:
        self.scenario_generator = scenario_generator

    def generate_trajectories(
        self,
        num_seed_scenarios: int,
        generation_config: GenerationConfig,
        num_parallel_vms: int = 1,
        execution_config: ExecutionConfig = ExecutionConfig(),
        task_config_base_dir: str = "external_data/osworld_evaluation_examples",
        trajectory_base_dir: str = "external_data/osworld-verified/jedi-7b-4o-15steps/jedi-7b-4o-15steps",
        result_base_dir: str = "./perturbation_results",
    ) -> List[GenerationResult]:
        """Generate trajectories with perturbation injection"""
        total_trajectories = (
            generation_config.num_invariance_scenarios
            + generation_config.num_distractor_scenarios
            + generation_config.num_negative_scenarios
        ) * generation_config.num_difficulty_levels

        logger.info(
            f"Generating trajectories for {num_seed_scenarios} seed scenarios..."
            f"Total trajectories: {total_trajectories}"
            f"- {generation_config.num_invariance_scenarios} invariance"
            f"- {generation_config.num_distractor_scenarios} distractor"
            f"- {generation_config.num_negative_scenarios} negative"
            f"- {generation_config.num_difficulty_levels} levels of difficulty"
        )
        seed_trajectories = self.load_seed_trajectories(task_config_base_dir, trajectory_base_dir)

        scenario_specs = self.scenario_generator.generate_scenarios(
            seed_trajectories, generation_config, result_base_dir
        )

        with Manager() as manager:
            shared_results = manager.list()
            scenario_queue = manager.Queue()

            for scenario_spec in scenario_specs:
                scenario_queue.put(scenario_spec)

            processes = []
            for i in range(num_parallel_vms):
                execution_engine = ParallelExecutionEngine(execution_config)
                p = Process(
                    target=execution_engine.run_vm_tasks,
                    args=(scenario_queue, shared_results),
                    name=f"PerturbationProcess-{i + 1}",
                )
                p.daemon = True
                p.start()
                processes.append(p)
                logger.info(f"Started process {p.name} with PID {p.pid}")

            try:
                while True:
                    alive_count = sum(1 for p in processes if p.is_alive())
                    if scenario_queue.empty():
                        logger.info("All tasks finished.")
                        break
                    if alive_count == 0:
                        logger.error("All processes died, exiting.")
                        break
                    time.sleep(5)

                for p in processes:
                    p.join()

            except KeyboardInterrupt:
                logger.info("Main process received KeyboardInterrupt.")
                raise
            except Exception as e:
                logger.error(f"Unexpected error: {e}", exc_info=True)
                raise

            results = list(shared_results)
            logger.info(
                f"Average result: {sum(r.result_score for r in results) / len(results) if results else 0}"
            )
            return results

    def load_seed_trajectories(self, config_base_dir: str, trajectory_base_dir: str) -> List[SeedTrajectory]:
        """Load seed trajectories from task configs and existing trajectories"""
        from pathlib import Path

        seed_trajectories: List[SeedTrajectory] = []
        config_path = Path(config_base_dir)

        # Find all task config JSON files in the evaluation examples
        # Prefer an 'examples' subdirectory if present (works for both
        # src/OSWorld/evaluation_examples and external_data/osworld_evaluation_examples)
        examples_dir = config_path / "examples" if (config_path / "examples").exists() else config_path

        if not examples_dir.exists():
            raise FileNotFoundError(f"Examples directory not found: {examples_dir}")

        # Optionally load failure intersection filters to filter seeds
        failure_ids, failure_filter_keys = self._try_load_failure_intersection()

        # Get all app directories (chrome, gimp, etc.)
        app_dirs = [d for d in examples_dir.iterdir() if d.is_dir()]

        for app_dir in app_dirs:
            app_name = app_dir.name
            logger.info(f"Loading trajectories for app: {app_name}")

            # Find all JSON config files in this app directory
            config_files = list(app_dir.glob("*.json"))

            for config_file in config_files:
                try:
                    with open(config_file, "r", encoding="utf-8") as f:
                        task_config = json.load(f)

                    # Verify required fields
                    if not all(
                        field in task_config for field in ["id", "instruction", "config", "evaluator"]
                    ):
                        logger.warning(f"Skipping {config_file.name} - missing required fields")
                        continue

                    # Construct trajectory path based on the task ID
                    task_id = task_config["id"]
                    task_trajectory_dir = os.path.join(trajectory_base_dir, app_name, task_id)

                    # Verify trajectory directory exists
                    if not os.path.exists(task_trajectory_dir):
                        logger.warning(f"Trajectory directory not found: {task_trajectory_dir}")
                        continue

                    # Verify traj.jsonl exists
                    traj_file = os.path.join(task_trajectory_dir, "traj.jsonl")
                    if not os.path.exists(traj_file):
                        logger.warning(f"Trajectory file not found: {traj_file}")
                        continue

                    # Skip if not in failure intersection (when available)
                    task_id = task_config["id"]
                    if failure_ids is not None and task_id not in failure_ids:
                        continue
                    key = (app_name, task_config["instruction"])  # (task_type, instruction)
                    if failure_filter_keys is not None and key not in failure_filter_keys:
                        continue

                    # Create seed trajectory with trajectory path
                    seed_trajectory = SeedTrajectory(
                        task_type=app_name,
                        task_instruction=task_config["instruction"],
                        config=task_config,
                        gt_actions_file_path=traj_file,
                        gt_actions=None,
                    )

                    seed_trajectories.append(seed_trajectory)
                    logger.debug(f"Loaded trajectory: {task_id}")

                except (json.JSONDecodeError, KeyError, OSError) as e:
                    logger.error(f"Error loading {config_file.name}: {e}")
                    continue

        # Apply per-type sampling caps
        rng = random.Random(int(os.environ.get("SEED_TRAJECTORIES_SEED", str(DEFAULT_SEED_TRAJECTORIES_SEED))))
        counts_by_type = dict(DEFAULT_SAMPLES_REQUIRED)

        # Strictly validate availability before sampling
        self._assert_required_counts_available(seed_trajectories, counts_by_type)
        selected = self._sample_seed_trajectories_by_type(seed_trajectories, counts_by_type, rng)

        # Log distribution
        dist: Dict[str, int] = {}
        for st in selected:
            dist[st.task_type] = dist.get(st.task_type, 0) + 1
        logger.info(
            "Selected %d seed trajectories after per-type caps: %s",
            len(selected),
            ", ".join(f"{k}={v}" for k, v in sorted(dist.items())),
        )
        return selected

    def load_seed_scenarios(self, config_base_dir: str, trajectory_base_dir: str) -> List[dict]:
        """Back-compat wrapper that returns a list of dicts instead of SeedTrajectory.

        This interfaces previous callers expecting `List[Dict[str, Any]]` while
        reusing the typed `SeedTrajectory` loader.
        """
        seed_trajectories = self.load_seed_trajectories(config_base_dir, trajectory_base_dir)
        return [asdict(st) for st in seed_trajectories]

    # ---------------------------
    # Internals / Helpers
    # ---------------------------

    def _read_failed_task_ids(self, path: str) -> Set[str]:
        """Read failure IDs from either {app_type:[ids...]} JSON or JSONL of objects.

        Returns a set of IDs.
        """
        try:
            text = open(path, "r", encoding="utf-8").read()
        except FileNotFoundError:
            return set()

        # Try object mapping format first
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                ids: Set[str] = set()
                for v in obj.values():
                    if isinstance(v, list):
                        for task_id in v:
                            if isinstance(task_id, str):
                                ids.add(task_id)
                return ids
        except json.JSONDecodeError:
            pass

        # Fallback to JSONL
        ids: Set[str] = set()
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            task_id = rec.get("id")
            if isinstance(task_id, str):
                ids.add(task_id)
        return ids

    def _read_failed_task_pairs(self, path: str) -> Set[Tuple[str, str]]:
        """Read JSONL file with fields {"instruction", "task_type"}.

        Returns a set of (task_type, instruction) tuples.
        """
        keys: Set[Tuple[str, str]] = set()
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        task_type = obj.get("task_type")
                        instruction = obj.get("instruction")
                        if isinstance(task_type, str) and isinstance(instruction, str):
                            keys.add((task_type, instruction))
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            return set()
        return keys

    def _try_load_failure_intersection(self) -> tuple[Set[str] | None, Set[Tuple[str, str]] | None]:
        """Attempt to load and intersect failure sets from known files.

        Returns a tuple (ids_intersection, pairs_intersection). Exactly one of the
        values will be non-None when both files are found and share a compatible
        format. If either file is missing, returns (None, None).
        """
        o3_path = os.environ.get("O3_FAILURES_PATH", DEFAULT_O3_FAILURES_PATH)
        uitars_path = os.environ.get("UITARS_FAILURES_PATH", DEFAULT_UITARS_FAILURES_PATH)

        o3_exists = os.path.exists(o3_path)
        u_exists = os.path.exists(uitars_path)
        if not (o3_exists and u_exists):
            logger.info(
                "Failure lists not found (o3=%s, uitars=%s). Loading all seeds.",
                o3_exists,
                u_exists,
            )
            return None, None

        # Prefer id-based if available in both
        o3_ids = self._read_failed_task_ids(o3_path)
        u_ids = self._read_failed_task_ids(uitars_path)
        if o3_ids and u_ids:
            ids_intersection = o3_ids.intersection(u_ids)
            logger.info(
                "Using id-based failure intersection: o3=%d, uitars=%d, intersection=%d",
                len(o3_ids),
                len(u_ids),
                len(ids_intersection),
            )
            return ids_intersection, None

        # Fallback to pair-based (task_type,instruction)
        o3_pairs = self._read_failed_task_pairs(o3_path)
        u_pairs = self._read_failed_task_pairs(uitars_path)
        if o3_pairs and u_pairs:
            intersection = o3_pairs.intersection(u_pairs)
            logger.info(
                "Using pair-based failure intersection: o3=%d, uitars=%d, intersection=%d",
                len(o3_pairs),
                len(u_pairs),
                len(intersection),
            )
            return None, intersection

        logger.info("Failure lists present but unrecognized/empty; loading all seeds.")
        return None, None

    def _log_insufficient_seeds(self, available: int, requested: int) -> None:
        if available < requested:
            logger.info(
                "Requested %d seed scenarios but only %d available after filtering; using all",
                requested,
                available,
            )

    def _sample_seed_trajectories_by_type(
        self,
        candidates: List[SeedTrajectory],
        counts_by_type: Dict[str, int],
        rng: random.Random,
    ) -> List[SeedTrajectory]:
        """Sample up to N per task_type from candidates.

        - If a type isn't in counts_by_type, take 0.
        - If available < requested, take all available.
        """
        grouped: Dict[str, List[SeedTrajectory]] = {}
        for st in candidates:
            grouped.setdefault(st.task_type, []).append(st)

        selected: List[SeedTrajectory] = []
        for task_type, group in grouped.items():
            n = counts_by_type.get(task_type, 0)
            if n <= 0:
                continue
            if len(group) <= n:
                selected.extend(group)
            else:
                selected.extend(rng.sample(group, n))
        return selected

    # ---------------------------
    # Validation
    # ---------------------------

    def _assert_required_counts_available(
        self,
        candidates: List[SeedTrajectory],
        counts_by_type: Dict[str, int],
    ) -> None:
        """Raise if any required per-type count cannot be met by candidates.

        Checks availability AFTER failure-intersection filtering.
        """
        # Count available per type
        available: Dict[str, int] = {}
        for st in candidates:
            available[st.task_type] = available.get(st.task_type, 0) + 1

        shortages: Dict[str, Tuple[int, int]] = {}
        for task_type, required in counts_by_type.items():
            if required <= 0:
                continue
            have = available.get(task_type, 0)
            if have < required:
                shortages[task_type] = (have, required)

        if shortages:
            details = ", ".join(f"{t} {have}/{req}" for t, (have, req) in sorted(shortages.items()))
            msg = (
                "Insufficient seed trajectories after filtering; cannot satisfy per-type requirements: "
                + details
            )
            raise RuntimeError(msg)


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

    # Initialize components
    scenario_generator = ScenarioGenerator()
    orchestrator = TrajectoryGenerationOrchestrator(scenario_generator)

    # Example usage
    execution_config = ExecutionConfig(
        # VM/Provider settings
        # path_to_vm="/Users/lockewang/FIG/OSWorld/vmware_vm_data/Ubuntu0/Ubuntu0.vmx",
        path_to_vm=None,
        provider_name="aws",
        region=os.environ["AWS_REGION"],
        snapshot_name=os.environ["AWS_SNAPSHOT_NAME"],
        # Environment settings
        headless=True,
        action_space="pyautogui",
        observation_type="screenshot",
        screen_size=(1920, 1080),
        os_type="Ubuntu",
        client_password="osworld-public-evaluation",
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

    # Test configuration
    task_config_base_dir = "src/OSWorld/evaluation_examples"
    trajectory_base_dir = "external_data/osworld-verified/jedi-7b-4o-15steps/jedi-7b-4o-15steps"
    result_base_dir = "/opt/manifold/results"

    results = orchestrator.generate_trajectories(
        num_seed_scenarios=2,
        generation_config=GenerationConfig(
            num_invariance_scenarios=1,
            num_distractor_scenarios=1,
            num_negative_scenarios=0,
            num_difficulty_levels=1,
        ),
        num_parallel_vms=1,
        execution_config=execution_config,
        task_config_base_dir=task_config_base_dir,
        trajectory_base_dir=trajectory_base_dir,
        result_base_dir=result_base_dir,
    )

    return results


if __name__ == "__main__":
    main()
