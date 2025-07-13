from typing import List, Dict, Any
import json
from .data_ingestion import DatasetLoader
from .model_adaptation import ModelAdapter
from .output_processing import OutputProcessor
from .utils import logger, MetricCalculator
from ..data.dataset import Dataset  # Integrate with Magma's data structures

class EvaluationPipeline:
    """
    Core orchestration class for zero-shot evaluations.
    Integrates with data and modeling components for seamless workflow.
    Supports dependency injection for modularity and testing.
    """

    def __init__(
        self,
        dataset_loader: DatasetLoader,
        model_adapter: ModelAdapter,
        output_processor: OutputProcessor,
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
            dataset: Dataset = self.dataset_loader.get_processed_dataset(dataset_path)

            results = []
            for item in dataset:
                input_data = self.model_adapter.prepare_input(item)
                raw_output = self.model_adapter.infer(input_data)
                processed_output = self.output_processor.process_output(raw_output, item)
                results.append({
                    'ground_truth': item.get('ground_truth'),
                    'prediction': processed_output
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