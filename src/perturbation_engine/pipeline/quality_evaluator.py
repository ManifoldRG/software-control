"""
QualityEvaluator: Wrapper around quality LLM
Clean interface for trajectory quality evaluation
"""

import logging
from typing import List

from perturbation_engine.pipeline.data_models import GeneratedTrajectory, ScenarioSpec
from perturbation_engine.pipeline.llm_services import QualityLLM


class QualityEvaluator:
    """Wrapper around quality LLM for trajectory evaluation"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.quality_llm = QualityLLM()

    def evaluate_trajectories(
        self, generated_trajectories: List[GeneratedTrajectory], scenario_specs: List[ScenarioSpec]
    ) -> List[float]:
        """Evaluate quality of generated trajectories"""

        self.logger.info(f"Evaluating {len(generated_trajectories)} trajectories")

        quality_scores = []
        scenario_spec_map = {spec.scenario_id: spec for spec in scenario_specs}

        for trajectory in generated_trajectories:
            try:
                scenario_spec = scenario_spec_map.get(trajectory.scenario_spec_id)
                if not scenario_spec:
                    self.logger.warning(f"No scenario spec found for {trajectory.scenario_spec_id}")
                    quality_scores.append(0.0)
                    continue

                # Evaluate trajectory quality
                quality_score = self.quality_llm.evaluate_trajectory_quality(trajectory, scenario_spec)
                quality_scores.append(quality_score)

                self.logger.info(f"Trajectory {trajectory.trajectory_id} quality: {quality_score}")

            except Exception as e:
                self.logger.error(f"Error evaluating trajectory {trajectory.trajectory_id}: {e}")
                quality_scores.append(0.0)

        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        self.logger.info(f"Average quality score: {avg_quality:.3f}")

        return quality_scores

    def evaluate_trajectory_quality(
        self, trajectory: GeneratedTrajectory, scenario_spec: ScenarioSpec
    ) -> float:
        """Evaluate quality of a single trajectory"""

        try:
            quality_score = self.quality_llm.evaluate_trajectory_quality(trajectory, scenario_spec)

            self.logger.info(f"Trajectory {trajectory.trajectory_id} quality: {quality_score}")
            return quality_score

        except Exception as e:
            self.logger.error(f"Error evaluating trajectory {trajectory.trajectory_id}: {e}")
            return 0.0

    def evaluate_single_trajectory(
        self, trajectory: GeneratedTrajectory, scenario_spec: ScenarioSpec
    ) -> float:
        """Evaluate quality of a single trajectory (alias for evaluate_trajectory_quality)"""
        return self.evaluate_trajectory_quality(trajectory, scenario_spec)
