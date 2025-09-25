"""
Unified Generator: Environment-First Curriculum-Based Trajectory Generation
Eliminates redundancy between static and curriculum generation
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from perturbation_engine.data_types import (
    EnvironmentState,
    ExecutionConfig,
    GenerationResult,
    ScenarioSpec,
    SeedTrajectory,
)
from perturbation_engine.pipeline.perturbation_desktop_env import PerturbationDesktopEnv
from perturbation_engine.pipeline.shared_execution_engine import SharedExecutionEngine
from perturbation_engine.simple_llm_orchestra import SimpleLLMOrchestra


@dataclass
class CurriculumPlan:
    """Curriculum plan based on environment analysis"""

    levels: List[Dict[str, Any]]
    focus_areas: List[str]
    invariant_targets: List[str]
    task_complexity: str
    recommended_perturbations: List[str]


class CurriculumPlanner:
    """Plans curriculum based on environment observation for invariant learning"""

    def __init__(self, llm_orchestra: SimpleLLMOrchestra = None):
        self.llm_orchestra = llm_orchestra or SimpleLLMOrchestra()
        self.logger = logging.getLogger(__name__)

    def plan_curriculum(self, env_state: EnvironmentState, seed_trajectory: SeedTrajectory) -> CurriculumPlan:
        """Create curriculum plan based on environment analysis"""

        # Analyze task complexity from environment TODO: fix the DOM tree string indexing here
        task_analysis = self._analyze_task_complexity(env_state)

        # Determine curriculum levels based on task
        curriculum_levels = self._determine_curriculum_levels(task_analysis)

        # Identify focus areas for invariant learning TODO: fix this
        focus_areas = self._identify_focus_areas(env_state)

        # Identify invariant targets TODO: fix this
        invariant_targets = self._identify_invariant_targets(env_state)

        return CurriculumPlan(
            levels=curriculum_levels,
            focus_areas=focus_areas,
            invariant_targets=invariant_targets,
            task_complexity=task_analysis.get("complexity", "medium"),
            recommended_perturbations=task_analysis.get("perturbations", []),
        )

    def _analyze_task_complexity(self, env_state: EnvironmentState) -> Dict[str, Any]:
        """Analyze task complexity from DOM and accessibility tree"""
        prompt = f"""
        Analyze this GUI task for curriculum planning:

        DOM Tree: {env_state.dom_tree[:2000]}
        A11Y Tree: {env_state.a11y_tree[:1000]}
        Task: {env_state.task_instruction}
        App Type: {env_state.app_type}

        Return JSON analysis:
        {{
            "complexity": "easy" | "medium" | "hard",
            "ui_elements": ["buttons", "forms", "navigation"],
            "interaction_types": ["click", "type", "select"],
            "perturbations": ["theme_change", "layout_shift", "text_variation"],
            "invariant_features": ["button_functionality", "form_validation"],
            "learning_focus": ["visual_robustness", "functional_consistency"]
        }}

        Focus on identifying:
        1. UI complexity (number of elements, interaction types)
        2. Task criticality (form submission, navigation, data entry)
        3. Invariant features that should remain consistent
        4. Areas where perturbations can improve learning
        """

        response = self.llm_orchestra.context_simplifier.call_llm(prompt)
        return response

    def _determine_curriculum_levels(self, task_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Determine curriculum levels based on task analysis"""
        complexity = task_analysis.get("complexity", "medium")
        # perturbations = task_analysis.get("perturbations", [])

        if complexity == "easy":
            return [
                {
                    "level": "easy",
                    "intensity": 0.2,
                    "perturbations": ["text_variation", "color_change"],
                    "count": 3,
                    "focus": "basic_robustness",
                },
                {
                    "level": "medium",
                    "intensity": 0.4,
                    "perturbations": ["theme_change", "layout_shift"],
                    "count": 2,
                    "focus": "visual_consistency",
                },
            ]
        elif complexity == "medium":
            return [
                {
                    "level": "easy",
                    "intensity": 0.3,
                    "perturbations": ["text_variation", "color_change"],
                    "count": 2,
                    "focus": "basic_robustness",
                },
                {
                    "level": "medium",
                    "intensity": 0.5,
                    "perturbations": ["theme_change", "layout_shift", "size_adjust"],
                    "count": 3,
                    "focus": "visual_consistency",
                },
                {
                    "level": "hard",
                    "intensity": 0.7,
                    "perturbations": ["complex_layout", "multiple_themes", "content_swap"],
                    "count": 2,
                    "focus": "functional_robustness",
                },
            ]
        else:  # hard
            return [
                {
                    "level": "medium",
                    "intensity": 0.4,
                    "perturbations": ["theme_change", "layout_shift"],
                    "count": 2,
                    "focus": "visual_consistency",
                },
                {
                    "level": "hard",
                    "intensity": 0.6,
                    "perturbations": ["complex_layout", "multiple_themes"],
                    "count": 3,
                    "focus": "functional_robustness",
                },
                {
                    "level": "expert",
                    "intensity": 0.8,
                    "perturbations": ["full_redesign", "accessibility_changes", "complex_interactions"],
                    "count": 2,
                    "focus": "invariant_learning",
                },
            ]

    def _identify_focus_areas(self, env_state: EnvironmentState) -> List[str]:
        """Identify focus areas for invariant learning"""
        # Analyze DOM structure to identify key UI patterns
        focus_areas = []

        if "form" in env_state.dom_tree.lower():
            focus_areas.append("form_interaction")
        if "button" in env_state.dom_tree.lower():
            focus_areas.append("button_functionality")
        if "navigation" in env_state.dom_tree.lower():
            focus_areas.append("navigation_consistency")
        if "input" in env_state.dom_tree.lower():
            focus_areas.append("input_validation")

        return focus_areas

    def _identify_invariant_targets(self, env_state: EnvironmentState) -> List[str]:
        """Identify invariant targets that should remain consistent"""
        # Use LLM to identify what should remain invariant
        prompt = f"""
        Identify invariant features in this GUI that should remain consistent during perturbations:

        DOM: {env_state.dom_tree[:1000]}
        Task: {env_state.task_instruction}

        Return JSON:
        {{
            "invariant_targets": [
                "button_functionality",
                "form_validation",
                "navigation_structure",
                "data_integrity"
            ]
        }}
        """

        response = self.llm_orchestra.context_simplifier.call_llm(prompt)
        return response.get("invariant_targets", [])


class UnifiedGenerator:
    """Unified generator for environment-first curriculum-based trajectory generation"""

    def __init__(self, execution_config: ExecutionConfig = None):
        self.execution_config = execution_config or ExecutionConfig()
        self.logger = logging.getLogger(__name__)
        self.llm_orchestra = SimpleLLMOrchestra()
        self.curriculum_planner = CurriculumPlanner(self.llm_orchestra)

    def generate_trajectories(
        self,
        seed_trajectory: SeedTrajectory,
        num_parallel_vms: int = 1,
        result_base_dir: str = "./curriculum_results",
    ) -> List[GenerationResult]:
        """Generate trajectories using environment-first curriculum approach - simplified"""

        self.logger.info(f"Starting environment-first curriculum generation for {seed_trajectory.task_type}")

        try:
            # Initialize environment and extract first observation
            env = self._initialize_environment()
            env_state = self._extract_first_observation(env, seed_trajectory)

            # Generate curriculum based on environment observation
            curriculum_plan = self.curriculum_planner.plan_curriculum(env_state, seed_trajectory)

            self.logger.info(f"Generated curriculum with {len(curriculum_plan.levels)} levels")
            self.logger.info(f"Focus areas: {curriculum_plan.focus_areas}")
            self.logger.info(f"Invariant targets: {curriculum_plan.invariant_targets}")

            # Generate scenarios for each curriculum level
            all_scenarios = self._generate_curriculum_scenarios(
                seed_trajectory, curriculum_plan, result_base_dir
            )

            # Execute scenarios
            results = self._execute_scenarios(env, all_scenarios, num_parallel_vms)

            self.logger.info(f"Generated {len(results)} trajectories")
            return results

        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            return []

    def _initialize_environment(self) -> PerturbationDesktopEnv:
        """Initialize environment for observation"""
        return PerturbationDesktopEnv(
            path_to_vm=self.execution_config.path_to_vm,
            action_space=self.execution_config.action_space,
            provider_name=self.execution_config.provider_name,
            region=self.execution_config.region,
            snapshot_name=self.execution_config.snapshot_name,
            screen_size=self.execution_config.screen_size,
            headless=self.execution_config.headless,
            os_type=self.execution_config.os_type,
            require_a11y_tree=self.execution_config.require_a11y_tree,
            require_terminal=self.execution_config.require_terminal,
            enable_proxy=self.execution_config.enable_proxy,
            client_password=self.execution_config.client_password,
            cache_dir=self.execution_config.cache_dir,
            chromium_port=self.execution_config.chromium_port,
        )

    def _extract_first_observation(
        self, env: PerturbationDesktopEnv, seed_trajectory: SeedTrajectory
    ) -> EnvironmentState:
        """Extract computer states and task context from first observation - simplified approach"""
        from perturbation_engine.simple_app_state_extractor import SimpleAppStateExtractor

        # Use simple app state extractor
        extractor = SimpleAppStateExtractor()
        app_state = extractor.extract_app_state(
            env, seed_trajectory.task_type, seed_trajectory.task_instruction
        )

        # Convert to EnvironmentState
        return EnvironmentState(
            dom_tree=f"<html><body><div class='{app_state.current_view}'>{app_state.task_context}</div></body></html>",
            a11y_tree=f"app: {app_state.app_type}, view: {app_state.current_view}",
            app_type=app_state.app_type,
            current_url=f"app://{app_state.app_type}",
            window_state={"width": 1920, "height": 1080},
            task_instruction=seed_trajectory.task_instruction,
        )

    def _detect_app_type(self, task_type: str) -> str:
        """Detect application type from task type"""
        if task_type == "chrome":
            return "browser"
        elif task_type in [
            "libreoffice_calc",
            "libreoffice_writer",
            "libreoffice_impress",
            "gimp",
            "vlc",
            "vs_code",
        ]:
            return "desktop_app"
        else:
            return "browser"

    def _generate_curriculum_scenarios(
        self, seed_trajectory: SeedTrajectory, curriculum_plan: CurriculumPlan, result_base_dir: str
    ) -> List[ScenarioSpec]:
        """Generate scenarios with LLM variations for each curriculum level"""

        all_scenarios = []
        scenario_count = 0

        # Generate scenarios for each curriculum level (without LLM variations yet)
        for level in curriculum_plan.levels:
            self.logger.info(f"Creating scenarios for curriculum level: {level['level']}")

            # Create scenarios that will use LLM orchestra at runtime
            for i in range(level["count"]):
                scenario = ScenarioSpec(
                    scenario_id=f"curriculum_{level['level']}_{i}",
                    task_id=seed_trajectory.config.get("id", "unknown"),
                    task_type=seed_trajectory.task_type,
                    scenario_type="curriculum_generated",
                    difficulty_level=self._level_to_difficulty(level["level"]),
                    seed_trajectory=seed_trajectory,
                    trajectory_file_path=seed_trajectory.gt_actions_file_path,
                    perturbation_scenario_class="CurriculumGeneratedScenario",
                    intensity=level["intensity"],
                    perturbation_count=1,
                    parameters={
                        "curriculum_level": level["level"],
                        "intensity": level["intensity"],
                        "perturbations": level["perturbations"],
                        "focus": level["focus"],
                        "use_llm_orchestra": True,  # Use LLM orchestra at runtime!
                        "app_type": curriculum_plan.focus_areas[0]
                        if curriculum_plan.focus_areas
                        else "browser",
                        "current_view": "unknown",
                    },
                    result_dir=f"{result_base_dir}/curriculum_{level['level']}_{i}",
                    seed_index=0,
                    scenario_count=scenario_count,
                )
                all_scenarios.append(scenario)
                scenario_count += 1

        self.logger.info(f"Generated {len(all_scenarios)} curriculum scenarios with LLM variations")
        return all_scenarios

    def _get_app_state_from_plan(self, curriculum_plan: CurriculumPlan):
        """Extract app state from curriculum plan"""
        from perturbation_engine.simple_app_state_extractor import AppState

        # Create basic app state from plan
        return AppState(
            app_type=curriculum_plan.focus_areas[0] if curriculum_plan.focus_areas else "unknown",
            current_view="unknown",
            key_elements=["button", "input", "menu"],
            task_context=curriculum_plan.invariant_targets[0]
            if curriculum_plan.invariant_targets
            else "unknown",
        )

    def _level_to_difficulty(self, level: str) -> int:
        """Convert curriculum level to difficulty level"""
        mapping = {"easy": 1, "medium": 2, "hard": 3, "expert": 4}
        return mapping.get(level, 2)

    def _execute_scenarios(self, scenarios: List, num_parallel_vms: int) -> List[GenerationResult]:
        """Execute scenarios using shared execution engine"""
        execution_engine = SharedExecutionEngine(self.execution_config)
        return execution_engine.execute_scenarios_parallel(scenarios, num_parallel_vms, "UnifiedCurriculum")
