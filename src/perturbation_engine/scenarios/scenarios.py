"""
Perturbation scenarios for different applications.
"""

from typing import Any, Dict, List

from perturbation_engine.scenarios.base_scenario import PerturbationScenario


class ChromePerturbationScenario(PerturbationScenario):
    """Chrome information retrieval perturbation scenario."""

    def apply_setup_perturbations(
        self,
        task_config: Dict[str, Any],
        perturbation_scenario: "PerturbationScenario",
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply setup perturbations for Chrome tasks."""
        # TODO: Implement Chrome-specific setup perturbations
        # - Change browser theme
        # - Modify task instruction
        # - Set up environment distractors
        return task_config

    def check_and_apply_runtime_perturbations(
        self,
        env: Any,
        perturbation_scenario: "PerturbationScenario",
        parameters: Dict[str, Any],
        step_idx: int,
        obs: Dict[str, Any],
        perturbation_log: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Get runtime perturbations for Chrome tasks."""
        try:
            nav_html = env.controller.get_interactable_html()
            if not nav_html:
                return {"applied": False, "error": "Could not extract page HTML"}

            js_code = env.controller.vlm_controller.get_ui_perturbation_js_code(nav_html, parameters)
            if not js_code:
                return {"applied": False, "error": "Could not generate JavaScript code"}

            success = env.controller.execute_js_on_page(js_code)

            return {
                "applied": success,
                "method": "gemini_ui_perturbation",
                "js_code": js_code,
                "parameters": parameters,
            }

        except Exception as e:
            self.logger.error(f"Error applying VLM perturbation: {e}")
            return {"applied": False, "error": str(e)}

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate Chrome scenario parameters."""
        return isinstance(parameters.get("num_components", 5), int)


def perturb_chrome_information_retrieval():
    """Legacy function for backward compatibility."""
    return ChromePerturbationScenario()


def perturb_gimp():
    pass


def perturb_libreoffice_calc():
    pass


def perturb_libreoffice_impress():
    pass


def perturb_libreoffice_writer():
    pass


def perturb_multi_apps():
    pass


def perturb_os():
    pass


def perturb_thunderbird():
    pass


def perturb_vlc():
    pass


def perturb_vs_code():
    pass
