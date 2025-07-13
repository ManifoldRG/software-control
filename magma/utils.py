import logging
from typing import List, Dict, Any, Callable
import abc

# Logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class MetricCalculator(abc.ABC):
    """
    Abstract base class for computing evaluation metrics.
    This allows for flexible metric definitions, injectable into the pipeline.
    Subclasses can implement specific metric sets (e.g., accuracy, F1).
    """

    @abc.abstractmethod
    def compute_metrics(self, results: List[Dict[str, Any]], metric_names: List[str]) -> Dict[str, Any]:
        """
        Compute metrics from results.
        metric_names specifies which metrics to compute.
        """
        pass

# Example concrete implementation:
# class BasicMetricCalculator(MetricCalculator):
#     def compute_metrics(self, results: List[Dict[str, Any]], metric_names: List[str]) -> Dict[str, Any]:
#         metrics = {}
#         if 'accuracy' in metric_names:
#             ... (compute accuracy)
#         return metrics

# Utility functions (add more as needed, e.g., image utils, text utils)
def register_metric(name: str) -> Callable:
    """
    Decorator to register custom metric functions.
    """
    def decorator(func: Callable):
        # Could maintain a registry of metrics
        return func
    return decorator 