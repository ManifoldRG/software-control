from typing import Any, Dict
from magma import Magma  # Assuming Magma is imported from the project's magma module
from .utils import logger

class ModelAdapter:
    """
    Class for adapting the Magma model for zero-shot evaluation on UI tasks.
    For zero-shot, this mainly involves prompt engineering and input formatting.
    """

    def __init__(self, model_path: str, device: str = 'cuda'):
        self.model = Magma.from_pretrained(model_path).to(device)
        logger.info(f"Loaded Magma model from {model_path} on {device}")

    def prepare_input(self, data_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare the input for the model, e.g., format prompt with image.
        """
        image = data_item.get('image')
        prompt = data_item.get('prompt', "Describe the UI action: ")
        # Assuming Magma takes image and text prompt
        input_data = {'image': image, 'prompt': prompt}
        return input_data

    def infer(self, input_data: Dict[str, Any]) -> Any:
        """
        Run inference on the adapted model.
        """
        output = self.model.generate(input_data)  # Adjust based on actual Magma API
        return output

    def adapt_for_task(self, task_config: Dict[str, Any]):
        """
        Apply any task-specific adaptations, e.g., adding a linear head if needed.
        For zero-shot, this might be a no-op or just set prompts.
        """
        logger.info(f"Adapting model for task: {task_config.get('task_name')}")
        # Implement if architectural changes are needed; for zero-shot, minimal
        pass