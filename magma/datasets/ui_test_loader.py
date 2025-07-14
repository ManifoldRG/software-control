from typing import List, Dict, Any
from PIL import Image
import base64
import io
from huggingface_hub import snapshot_download
from agents.ui_agent.util.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img
from ..data_ingestion import DatasetLoader
from ..utils import logger

class UITestDatasetLoader(DatasetLoader):
    def __init__(self, config: Dict[str, Any] = None):
        self.instruction = config.get('instruction') if config else None
        
        #For SoM annotation
        repo_id = "microsoft/OmniParser-v2.0"  # HF repo
        local_dir = "weights"  # Target local directory

        # Download the entire repository
        snapshot_download(repo_id=repo_id, local_dir=local_dir)
        print(f"Repository downloaded to: {local_dir}")

    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        if dataset_path.endswith('.json'):
            import json
            with open(dataset_path, 'r') as f:
                raw_data = json.load(f)
            dataset = []
            for item in raw_data:
                data_item = {
                    'image_path': item['image_path'],
                    'instruction': item['instruction'],
                    'ground_truth': item.get('ground_truth')
                }
                image = Image.open(data_item['image_path']).convert('RGB')
                data_item['image'] = image
                dataset.append(data_item)
            return dataset
        else:
            # Single image
            image = Image.open(dataset_path).convert('RGB')
            data_item = {'image': image}
            if self.instruction:
                data_item['instruction'] = self.instruction
            return [data_item]

    def preprocess(self, data_item: Dict[str, Any]) -> Dict[str, Any]:
        image = data_item['image']

        # Preprocessing steps from magma_inference_test.py
        box_overlay_ratio = image.size[0] / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }

        yolo_model = get_yolo_model(model_path='weights/icon_detect/model.pt')
        caption_model_processor = get_caption_model_processor(model_name='florence2', model_name_or_path='weights/icon_caption')

        ocr_bbox_rslt, is_goal_filtered = check_ocr_box(image, display_img=False, output_bb_format='xyxy', goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold': 0.9})
        text, ocr_bbox = ocr_bbox_rslt
        dino_labeled_img, label_coordinates, parsed_content_list = get_som_labeled_img(image, yolo_model, output_coord_in_ratio=False, ocr_bbox=ocr_bbox, ocr_text=text, caption_model_processor=caption_model_processor)

        annotated_image = Image.open(io.BytesIO(base64.b64decode(dino_labeled_img)))

        data_item['annotated_image'] = annotated_image
        data_item['label_coordinates'] = label_coordinates
        data_item['parsed_content_list'] = parsed_content_list
        # Retain instruction and ground_truth
        logger.info('Preprocessed UI data item with annotations.')
        return data_item