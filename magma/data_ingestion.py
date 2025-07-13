import abc
from typing import List, Dict, Any
import os
import sys

# Add parent directory to path to allow direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from data.dataset import Dataset  # Integrate with Magma's existing data module for consistency
from magma.utils import logger

class DatasetLoader(abc.ABC):
    """
    Abstract base class for loading and preprocessing datasets.
    This abstraction ensures compatibility with data pipelines while allowing extension to new formats.
    Subclasses should implement load_dataset and preprocess.
    """

    @abc.abstractmethod
    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """
        Load the raw dataset from the given path.
        Returns a list of data items as dictionaries (e.g., {'image': ..., 'prompt': ..., 'ground_truth': ...}).
        Raise appropriate exceptions for invalid paths or formats.
        """
        pass

    @abc.abstractmethod
    def preprocess(self, data_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess a single data item.
        Apply transformations like image loading, text tokenization, or Magma-specific data formatting.
        """
        pass

    def get_processed_dataset(self, dataset_path: str) -> Dataset:
        """
        Load and preprocess the entire dataset, returning a Magma Dataset object for integration.
        This method ensures the output is pipeline-ready.
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")
        raw_dataset = self.load_dataset(dataset_path)
        processed = [self.preprocess(item) for item in raw_dataset]
        logger.info(f"Loaded and processed {len(processed)} items from {dataset_path}")
        return Dataset(processed)  # Wrap in Magma's Dataset for consistency

# Example subclass (commented; implement in separate files like evaluation/datasets/ui_loader.py):
# class UILoader(DatasetLoader):
#     def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
#         ... 
#     def preprocess(self, data_item: Dict[str, Any]) -> Dict[str, Any]]:
#         ... 