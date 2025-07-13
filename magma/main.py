import argparse
from typing import Type, List, Dict, Any
import json
import abc
from functools import lru_cache
import logging
import importlib  # For dynamic class loading

# Logger setup, integrated with Magma's potential logging if exists
logger = logging.getLogger("magma.evaluation")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# This file makes the evaluation directory a Python package.
# It exposes key classes for easy import across the Magma project.
from .data_ingestion import DatasetLoader
from .model_adaptation import ModelAdapter
from .output_processing import OutputProcessor
from .evaluation_pipeline import EvaluationPipeline
from .utils import MetricCalculator, load_config, logger
from ..data.dataset import Dataset  # Integrate with Magma's data structures
from ..magma.modeling_magma import Magma  # Directly integrate with Magma's core modeling

class MetricCalculator(abc.ABC):
    """
    Abstract base class for metric computation in Magma evaluations.
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

# Example concrete class (extendable):
# class DefaultMetricCalculator(MetricCalculator):
#     def __init__(self):
#         super().__init__()
#         self.register_metric('accuracy', self._compute_accuracy)
# 
#     def _compute_accuracy(self, results: List[Dict[str, Any]]) -> float:
#         ...

@lru_cache(maxsize=128)
def load_config(config_path: str) -> Dict[str, Any]:
    """Cached config loading for efficiency."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Add more Magma-integrated utils, e.g., image processing from magma.image_processing_magma

def instantiate_class(module_path: str, class_name: str, config: Dict = None) -> Any:
    """Factory to dynamically instantiate classes from config."""
    module = importlib.import_module(module_path)
    cls: Type = getattr(module, class_name)
    return cls(config) if config else cls()

def main():
    parser = argparse.ArgumentParser(description="Magma Zero-Shot Evaluation Framework")
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config (aligns with data_configs/ style).")
    args = parser.parse_args()

    config = load_config(args.config)
    try:
        # Dynamic instantiation for high flexibility
        loader_config = config['data_ingestion']
        loader = instantiate_class(loader_config['module'], loader_config['class'], loader_config.get('config'))

        adapter_config = config['model_adaptation']
        adapter = instantiate_class(adapter_config['module'], adapter_config['class'], adapter_config.get('config'))

        processor_config = config['output_processing']
        processor = instantiate_class(processor_config['module'], processor_config['class'], processor_config.get('config'))

        metric_config = config['metrics']
        metric_calc = instantiate_class(metric_config['module'], metric_config['class'], metric_config.get('config'))

        pipeline = EvaluationPipeline(loader, adapter, processor, metric_calc)

        pipeline.run_evaluation(
            dataset_path=config['dataset_path'],
            output_path=config.get('output_path', 'results.json'),
            task_config=config['task_config']
        )
    except Exception as e:
        logger.error("Pipeline initialization failed: %s", str(e))
        raise

if __name__ == "__main__":
    main() 