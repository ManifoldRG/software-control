from PIL import Image
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor
from agents.ui_agent.util.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img
from huggingface_hub import snapshot_download
import io
import base64
import time

dtype = torch.bfloat16
model = AutoModelForCausalLM.from_pretrained("microsoft/Magma-8B", trust_remote_code=True, torch_dtype=dtype)
processor = AutoProcessor.from_pretrained("microsoft/Magma-8B", trust_remote_code=True)
model.to("cuda")


repo_id = "microsoft/OmniParser-v2.0"  # HF repo
local_dir = "weights"  # Target local directory

snapshot_download(repo_id=repo_id, local_dir=local_dir)
print(f"Repository downloaded to: {local_dir}")


# Inference
image = Image.open("assets/images/mind2web.png").convert("RGB")

box_overlay_ratio = image.size[0] / 3200
draw_bbox_config = {
        'text_scale': 0.8 * box_overlay_ratio,
        'text_thickness': max(int(2 * box_overlay_ratio), 1),
        'text_padding': max(int(3 * box_overlay_ratio), 1),
        'thickness': max(int(3 * box_overlay_ratio), 1),
    }


yolo_model = get_yolo_model(model_path='weights/icon_detect/model.pt')
caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="weights/icon_caption")

ocr_bbox_rslt, is_goal_filtered = check_ocr_box(image, display_img = False, output_bb_format='xyxy', goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold':0.9})
text, ocr_bbox = ocr_bbox_rslt
dino_labeled_img, label_coordinates, parsed_content_list = get_som_labeled_img(image, yolo_model,output_coord_in_ratio=False, ocr_bbox=ocr_bbox, ocr_text=text, caption_model_processor = caption_model_processor)
#parsed_content_list = '\n'.join([f'icon {i}: ' + str(v) for i,v in enumerate(parsed_content_list)])
image = Image.open(io.BytesIO(base64.b64decode(dino_labeled_img)))
image.save('som_annotated.png')

'''convs = [
    {"role": "system", "content": """You are a meticulous AI assistant specializing in UI interaction analysis.
You will be provided with a screenshot of a user interface that has been annotated with numbered bounding boxes highlighting interactive elements such as buttons, icons, and text fields.
Your mission is to accurately identify the specific UI element that the user wishes to interact with, based on their request.

Your response must be precise and follow this format:
- Identify the bounding box index for the element.
- Provide the coordinates of the center of that bounding box.
- Include a brief description of the element.

Example response:
The user wants to click on the search bar. The bounding box index is 42, and the coordinates are [x, y]."""},
    {"role": "user", "content": "<image_start><image><image_end>\nIn this view I need to click seomwhere to start searching? Provide the coordinates and the mark index of the containing bounding box if applicable."},
]'''

convs = [
    {"role": "system", "content":"You are agent that can see, talk and act."},
    {"role": "user", "content": """<image_start><image><image_end>
Imagine that you are imitating humans doing web navigation for a task step by step. At each stage, you can see the webpage like humans by a screenshot and know the previous actions before the current step decided by yourself through recorded history. You need to decide on the following action to take. You can click an element with the mouse, select an option, or type text with the keyboard. The output format should be a dictionary like: 
"{"ACTION": "CLICK" or "TYPE" or "SELECT", "MARK": a numeric id, e.g., 5, "VALUE": a string value for the action if applicable, otherwise None}".
You are asked to complete the following task: Find all outdoor events this month in NYC. The previous actions you have taken: 

[span]  Special events -> CLICK
[DisclosureTriangle]  All locations -> CLICK
[li]  NYC -> CLICK
For your convenience, I have labeled the candidates with numeric marks and bounding boxes on the screenshot. What is the next action you would take?
"""
    }
]


start_time = time.time()
prompt = processor.tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)
inputs = processor(images=[image], texts=prompt, return_tensors="pt")
inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
inputs['image_sizes'] = inputs['image_sizes'].unsqueeze(0)
inputs = inputs.to("cuda").to(dtype)

generation_args = {
    "max_new_tokens": 10000,
    "temperature": 0.0,
    "do_sample": False,
    "use_cache": True,
    "num_beams": 1,
}

with torch.inference_mode():
    generate_ids = model.generate(**inputs, **generation_args)

generate_ids = generate_ids[:, inputs["input_ids"].shape[-1] :]
response = processor.decode(generate_ids[0], skip_special_tokens=True).strip()
end_time = time.time()

print("\nTotal time taken for inference: ")
print(end_time-start_time)

print(response)
