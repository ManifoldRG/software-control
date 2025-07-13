import argparse
from typing import Type, List, Dict, Any
import json
import abc
from functools import lru_cache
import logging
import importlib  # For dynamic class loading
import yaml

# Logger setup, integrated with Magma's potential logging if exists
logger = logging.getLogger("magma.evaluation")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

from .evaluation_pipeline import EvaluationPipeline
from .utils import MetricCalculator, load_config, logger



@lru_cache(maxsize=128)
def load_config(config_path: str) -> Dict[str, Any]:
    """Cached config loading for efficiency."""
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def instantiate_class(module_path: str, class_name: str, config: Dict = None) -> Any:
    """Dynamically instantiate classes from config."""
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