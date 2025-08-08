#!/usr/bin/env python3
"""
Test script for scene analyzers
"""

import logging
import sys
from pathlib import Path

from perturbation_engine.data.trajectory_data import (
    Action,
    Episode,
    Observation,
    Step,
)
from perturbation_engine.scene.analyzer import SceneAnalyzer


def create_test_data():
    """Create test data for development."""
    mhtml_path = Path(__file__).resolve().parent / "0daf1895-493d-4b9a-ba8a-ba6a65c23a21_after.mhtml"

    test_task = Episode(
        episode_id="test_episode_001",
        task="Click the button",
        steps=[
            Step(
                index=0,
                observation=Observation(mhtml=mhtml_path),
                action=Action(action_type="click", coordinates=(100, 100)),
            )
        ],
    )

    return mhtml_path, test_task


def test_scene_analyzer():
    """Test the scene analyzer."""
    logging.info("Testing scene analyzer...")

    analyzer = SceneAnalyzer()
    mhtml_path, task_episode = create_test_data()

    analysis = analyzer.analyze_scene(mhtml_path, task_episode.task)
    logging.info(f"""
Scene ID: {analysis.scene_id}
Elements found: {len(analysis.elements)}
Goal relevant elements: {len(analysis.goal_relevant_elements)}
Background elements: {len(analysis.background_elements)}
Plausibility score: {analysis.plausibility_score}
Solvability score: {analysis.solvability_score}
                 """)

    assert analysis.scene_id == mhtml_path.stem
    assert len(analysis.elements) >= 1
    # assert len(analysis.goal_relevant_elements) >= 1
    assert len(analysis.background_elements) >= 1
    assert 0.0 <= analysis.plausibility_score <= 1.0
    assert 0.0 <= analysis.solvability_score <= 1.0

    perturbations = analyzer.propose_configs(analysis)
    logging.info("Generated perturbations: %d", len(perturbations))
    assert len(perturbations) == len(analysis.background_elements), (
        "Number of perturbations not equal to number of background elements"
    )

    bg_selectors = {e.selector for e in analysis.background_elements}
    cfg_selectors = {p.target_selector for p in perturbations}
    assert cfg_selectors.issubset(bg_selectors), "Config selectors not subset of background selectors"

    sampled_1 = perturbations[0].sample_concrete(seed=42)
    css_1 = sampled_1.parameters.to_css()
    assert "background-color" in css_1, "Background color not in CSS"
    assert css_1["background-color"].startswith("hsl("), "Background color not in HSL format"

    sampled_2 = perturbations[0].sample_concrete(seed=42)
    css_2 = sampled_2.parameters.to_css()
    assert css_1 == css_2, "CSS not deterministic with same seed"

    return analysis, perturbations


def main():
    import os

    from perturbation_engine.logging import configure_logging

    os.environ.setdefault("PERTURB_ENGINE_LOG_COLOR", "1")
    os.environ.setdefault("PERTURB_ENGINE_LOG_LEVEL", "INFO")
    configure_logging()
    print("\n=== Scene Analyzers Test ===")

    try:
        analysis, perturbations = test_scene_analyzer()

        logging.info("\n✅ All tests completed successfully!")
        logging.info(f"Scene analyzer found {len(analysis.elements)} elements")

    except Exception as e:
        logging.error(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
