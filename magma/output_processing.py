import abc
from typing import Any, Dict
import os
import sys

# Add parent directory to path to allow direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from magma.utils import logger

class OutputProcessor(abc.ABC):
    """
    Abstract base class for processing raw model outputs.
    Transforms outputs into standardized formats for metric computation.
    Designed to be task-agnostic with subclasses for specific parsing needs.
    """

    @abc.abstractmethod
    def process_output(self, raw_output: Any, data_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process raw output (e.g., text generations) into evaluable structures.
        Use data_item for contextual parsing if required.
        Raise ValueError for unparseable outputs.
        """
        pass

# Example subclass (commented):
# class UIOutputProcessor(OutputProcessor):
#     def process_output(self, raw_output: Any, data_item: Dict[str, Any]) -> Dict[str, Any]:
#         ... (e.g., parse action strings) 