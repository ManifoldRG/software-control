"""
QualityEvaluator: Wrapper around quality LLM
Clean interface for trajectory quality evaluation
"""

import logging

from perturbation_engine.pipeline.data_models import GeneratedTrajectory, ScenarioSpec
from perturbation_engine.pipeline.llm_services import QualityLLM


class QualityEvaluator:
    """Wrapper around quality LLM for trajectory evaluation"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.quality_llm = QualityLLM()

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
