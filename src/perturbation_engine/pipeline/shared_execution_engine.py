"""
Shared execution engine for parallel trajectory generation
"""

import logging
import time
from multiprocessing import Manager, Process
from typing import List

from perturbation_engine.data_types import ExecutionConfig, GenerationResult
from perturbation_engine.pipeline.parallel_execution_engine import ParallelExecutionEngine
from perturbation_engine.simple_llm_orchestra import SimpleLLMOrchestra


class SharedExecutionEngine:
    """Shared execution engine for parallel trajectory generation"""

    def __init__(self, execution_config: ExecutionConfig = None):
        self.execution_config = execution_config or ExecutionConfig()
        self.logger = logging.getLogger(__name__)

    def execute_scenarios_parallel(
        self, scenarios: List, num_parallel_vms: int = 1, process_name_prefix: str = "ExecutionProcess"
    ) -> List[GenerationResult]:
        """Execute scenarios in parallel using shared queue management"""

        with Manager() as manager:
            shared_results = manager.list()
            scenario_queue = manager.Queue()

            # Add scenarios to queue
            for scenario in scenarios:
                scenario_queue.put(scenario)

            # Start parallel processes
            processes = []
            for i in range(num_parallel_vms):
                execution_engine = ParallelExecutionEngine(self.execution_config, SimpleLLMOrchestra())
                p = Process(
                    target=execution_engine.run_vm_tasks,
                    args=(scenario_queue, shared_results),
                    name=f"{process_name_prefix}-{i + 1}",
                )
                p.daemon = True
                p.start()
                processes.append(p)
                self.logger.info(f"Started process {p.name} with PID {p.pid}")

            # Wait for completion
            try:
                while True:
                    alive_count = sum(1 for p in processes if p.is_alive())
                    if scenario_queue.empty():
                        self.logger.info("All tasks finished.")
                        break
                    if alive_count == 0:
                        self.logger.error("All processes died, exiting.")
                        break
                    time.sleep(5)

                for p in processes:
                    p.join()

            except KeyboardInterrupt:
                self.logger.info("Execution interrupted.")
                raise

            results = list(shared_results)
            self.logger.info(f"Completed {len(results)} trajectories")
            return results
