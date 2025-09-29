#!/usr/bin/env python3
"""
Debug script to test scenario generation and trace the list vs dict issue
"""

import json
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from perturbation_engine.pipeline.data_models import (
    CurriculumConfig,
    SeedTrajectory,
)


def main():
    print("🔍 DEBUG: Starting scenario generation test")

    # Load the inputs.json data
    with open("inputs.json", "r") as f:
        inputs = json.load(f)

    print(f"🔍 DEBUG: Loaded {len(inputs)} inputs from inputs.json")

    # Test with the first input
    if inputs:
        input_data = inputs[0]
        print(f"🔍 DEBUG: Testing with first input: {input_data['app_type']}")

        # Create objects
        seed_trajectory = SeedTrajectory(**input_data["seed_trajectory"])
        curriculum_config = CurriculumConfig(**input_data["curriculum_config"])
        app_states = [input_data["app_state"]]

        print("🔍 DEBUG: Created objects successfully")
        print(f"🔍 DEBUG: seed_trajectory.task_instruction: {seed_trajectory.task_instruction}")
        print(f"🔍 DEBUG: app_states: {app_states}")

        # Import and test CurriculumLLM
        from perturbation_engine.pipeline.llm_services import CurriculumLLM

        llm = CurriculumLLM()
        print("🔍 DEBUG: Created CurriculumLLM instance")

        # This should trigger the debug prints
        try:
            scenarios = llm.generate_scenario_specs(seed_trajectory, app_states, curriculum_config)
            print(f"🔍 DEBUG: Generated {len(scenarios)} scenarios")
            for i, scenario in enumerate(scenarios):
                print(f"🔍 DEBUG: Scenario {i}: {scenario}")
        except Exception as e:
            print(f"🚨 DEBUG: Exception during scenario generation: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
