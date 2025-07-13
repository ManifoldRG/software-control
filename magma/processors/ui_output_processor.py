from typing import Any, Dict
import time
from ..output_processing import OutputProcessor
from ..utils import logger

class UIOutputProcessor(OutputProcessor):
    def process_output(self, raw_output: Any, data_item: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        # For this test, the output is the response string
        processed = {'response': raw_output}
        end_time = time.time()
        processed['inference_time'] = end_time - start_time
        logger.info(f'Response: {raw_output}')
        logger.info(f'Time taken: {processed["inference_time"]}')
        return processed 