"""
Control Refactored: Clean interfaces for VM manipulation
"""

from .clean_target_element_tracker import CleanTargetElementTracker, ElementIdentity
from .perturbation_controller import ManipulationResult, PerturbationController

__all__ = ["PerturbationController", "ManipulationResult", "CleanTargetElementTracker", "ElementIdentity"]
