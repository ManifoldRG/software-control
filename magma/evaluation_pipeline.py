from typing import List, Dict, Any
import json
import os
import sys

# Add parent directory to path to allow direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from magma.datasets.ui_test_loader import UITestDatasetLoader
from magma.adapters.magma_ui_adapter import MagmaUIAdapter
from magma.processors.ui_output_processor import UIOutputProcessor
from magma.utils import logger, MetricCalculator

class EvaluationPipeline:
    """
    Core orchestration class for zero-shot evaluations.
    Integrates with data and modeling components for seamless workflow.
    Supports dependency injection for modularity and testing.
    """

    def __init__(
        self,
        dataset_loader: UITestDatasetLoader,
        model_adapter: MagmaUIAdapter,
        output_processor: UIOutputProcessor,
        metric_calculator: MetricCalculator
    ):
        self.dataset_loader = dataset_loader
        self.model_adapter = model_adapter
        self.output_processor = output_processor
        self.metric_calculator = metric_calculator

    def run_evaluation(
        self,
        dataset_path: str,
        output_path: str,
        task_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute the full evaluation pipeline.
        Handles errors gracefully and logs progress.
        """
        try:
            self.model_adapter.adapt_for_task(task_config)
            dataset: List[Dict[str, Any]] = self.dataset_loader.get_processed_dataset(dataset_path)

            results = []
            for item in dataset:
                input_data, bbox_coordinates = self.model_adapter.prepare_input(item)
                raw_output, bbox_coordinates = self.model_adapter.infer(input_data, bbox_coordinates)
                processed_output = self.output_processor.process_output(raw_output, item)
                results.append({
                    'ground_truth': item.get('ground_truth'),
                    'prediction': processed_output,
                    'bbox_coordinates': bbox_coordinates
                })

            metrics = self.metric_calculator.compute_metrics(results, task_config.get('metrics', []))

            self.save_results(output_path, results, metrics)
            logger.info("Evaluation completed with metrics: %s", metrics)
            return {'metrics': metrics, 'num_items': len(results)}
        except Exception as e:
            logger.error("Evaluation failed: %s", str(e))
            raise

    def save_results(self, output_path: str, results: List[Dict[str, Any]], metrics: Dict[str, Any]):
        """
        Save results in JSON format, extensible to other formats (e.g., CSV).
        """
        with open(output_path, 'w') as f:
            json.dump({'results': results, 'metrics': metrics}, f, indent=4) 