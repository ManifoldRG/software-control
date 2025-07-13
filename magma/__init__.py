# This file makes the evaluation directory a Python package.
# It exposes key classes for easy import across the Magma project.
from .data_ingestion import DatasetLoader
from .model_adaptation import ModelAdapter
from .output_processing import OutputProcessor
from .evaluation_pipeline import EvaluationPipeline
from .utils import MetricCalculator, logger 