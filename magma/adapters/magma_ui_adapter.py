from typing import Any, Dict
import torch
from ...magma.modeling_magma import MagmaForCausalLM
from ...magma.processing_magma import MagmaProcessor
from ..model_adaptation import ModelAdapter
from ..utils import logger

class MagmaUIAdapter(ModelAdapter):
    def load_model(self, model_config: Dict[str, Any]) -> None:
        dtype = torch.bfloat16
        model = MagmaForCausalLM.from_pretrained(model_config['path'], trust_remote_code=True, torch_dtype=dtype)
        processor = MagmaProcessor.from_pretrained(model_config['path'], trust_remote_code=True)
        model.to(model_config.get('device', 'cuda'))
        self.model = {'model': model, 'processor': processor}

    def prepare_input(self, data_item: Dict[str, Any]) -> Dict[str, Any]:
        image = data_item['annotated_image']
        instruction = data_item.get('instruction', 'Imagine that you are imitating humans doing web navigation for a task step by step. At each stage, you can see the webpage like humans by a screenshot. You need to decide on the following action to take. You can click an element with the mouse, select an option, or type text with the keyboard. The output format should be a dictionary like: \n"{"ACTION": "CLICK" or "TYPE" or "SELECT", "MARK": a numeric id, e.g., 5, "VALUE": a string value for the action if applicable, otherwise None}"\nFor your convenience, I have labeled the candidates with numeric marks and bounding boxes on the screenshot. What is the next action you would take?')
        user_content = f'<image_start><image><image_end>\n{instruction}'
        convs = [
            {'role': 'system', 'content': 'You are agent that can see, talk and act.'},
            {'role': 'user', 'content': user_content}
        ]
        processor = self.model['processor']
        prompt = processor.tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=[image], texts=prompt, return_tensors='pt')
        inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
        inputs['image_sizes'] = inputs['image_sizes'].unsqueeze(0)
        inputs = inputs.to(self.model['model'].device).to(torch.bfloat16)
        return inputs

    def infer(self, input_data: Dict[str, Any]) -> Any:
        model = self.model['model']
        generation_args = {
            'max_new_tokens': 10000,
            'temperature': 0.0,
            'do_sample': False,
            'use_cache': True,
            'num_beams': 1,
        }
        with torch.inference_mode():
            generate_ids = model.generate(**input_data, **generation_args)
        processor = self.model['processor']
        generate_ids = generate_ids[:, input_data['input_ids'].shape[-1]:]
        response = processor.decode(generate_ids[0], skip_special_tokens=True).strip()
        return response

    def adapt_for_task(self, task_config: Dict[str, Any]):
        super().adapt_for_task(task_config)
        logger.info('Adapted Magma for UI inference test.') 