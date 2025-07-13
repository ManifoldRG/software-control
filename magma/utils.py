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
    Abstract base class for metric computation.
    Supports dynamic metric registration for flexibility.
    """

    def __init__(self):
        self.metric_registry: Dict[str, Callable] = {}

    def register_metric(self, name: str, func: Callable[[List[Dict[str, Any]]], Any]):
        """
        Register a custom metric function.
        """
        self.metric_registry[name] = func

    @abc.abstractmethod
    def compute_metrics(self, results: List[Dict[str, Any]], metric_names: List[str]) -> Dict[str, Any]:
        """
        Compute requested metrics using the registry.
        """
        metrics = {}
        for name in metric_names:
            if name in self.metric_registry:
                metrics[name] = self.metric_registry[name](results)
            else:
                logger.warning(f"Metric {name} not registered.")
        return metrics