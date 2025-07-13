from typing import Any, Dict
from magma import Magma 
from .utils import logger

# Import abc for abstract base class
import abc

class ModelAdapter(abc.ABC):
    """
    Abstract base class for adapting models for zero-shot evaluation on UI tasks.
    Subclasses must implement load_model, and may override other methods as needed.
    """

    def __init__(self, config: Dict[str, Any]):
        self.load_model(config)

    @abc.abstractmethod
    def load_model(self, config: Dict[str, Any]) -> None:
        """
        Load the model using the provided configuration.
        Set self.model appropriately.
        """
        pass

    @abc.abstractmethod
    def prepare_input(self, data_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare the input for the model, e.g., format prompt with image.
        """
        pass

    @abc.abstractmethod
    def infer(self, input_data: Dict[str, Any]) -> Any:
        """
        Run inference on the adapted model.
        """
        pass

    def adapt_for_task(self, task_config: Dict[str, Any]):
        """
        Apply any task-specific adaptations.
        """
        logger.info(f"Adapting model for task: {task_config.get('task_name')}")
        pass