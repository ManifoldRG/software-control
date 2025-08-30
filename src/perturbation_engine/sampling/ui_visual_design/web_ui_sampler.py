"""Simple perturbation applier for MHTML content."""

import logging
from pathlib import Path

from perturbation_engine.sampling.ui_visual_design.data_types import (
    ColorParams,
    PerturbationConfig,
    PerturbationResult,
)


class WebUISampler:
    """Applies CSS perturbations to MHTML content."""

    def __init__(self):
        self._logger = logging.getLogger(__name__)

    def apply_perturbations(self, mhtml_file: Path, configs: list[PerturbationConfig]) -> PerturbationResult:
        """Apply perturbations and return result."""
        content = mhtml_file.read_text(encoding="utf-8")

        # Apply each config
        for config in configs:
            content = self._apply_config(content, config)

        # Create result
        perturbed_mhtml = mhtml_file.with_name(f"{mhtml_file.stem}_perturbed.mhtml")
        perturbed_mhtml.write_text(content, encoding="utf-8")

        return PerturbationResult(
            original_mhtml=mhtml_file,
            perturbed_mhtml=perturbed_mhtml,
            applied_perturbations=configs,
        )

    def _apply_config(self, content: str, config: PerturbationConfig) -> str:
        """Apply single config to content."""
        if isinstance(config.parameters, ColorParams):
            return self._inject_css(content, config.target_selector, config.parameters.to_css())

        self._logger.warning("Unsupported parameter type: %s", type(config.parameters))
        return content

    def _inject_css(self, content: str, selector: str, css_props: dict[str, str]) -> str:
        """Inject CSS rule into MHTML content."""
        if not css_props:
            return content

        # Create CSS rule
        props = "; ".join(f"{k}: {v}" for k, v in css_props.items())
        css_rule = f"{selector} {{ {props}; }}"
        style_tag = f"<style>{css_rule}</style>"

        # Inject into content
        if "</head>" in content:
            return content.replace("</head>", f"{style_tag}</head>")
        else:
            return style_tag + content
