import time

import psutil
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def check_system_resources():
    """Check available system resources"""
    print("=== System Resources ===")
    print(f"CPU cores: {psutil.cpu_count()}")
    print(f"Memory total: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    print(f"Memory available: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    print(f"Memory used: {psutil.virtual_memory().percent:.1f}%")

    print("\n=== PyTorch MPS Support ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")

    if torch.backends.mps.is_available():
        print("✓ MPS (Metal) is available for GPU acceleration")
    else:
        print("⚠ MPS not available, will use CPU")


def test_uitars_apple_silicon(model_path="ByteDance-Seed/UI-TARS-1.5-7B"):
    """Test optimized for Apple Silicon"""
    print("\n=== Testing UI-TARS on Apple Silicon ===")
    print(f"Model path: {model_path}")

    # Check resources first
    check_system_resources()

    try:
        # Determine device
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print(f"Using MPS device: {device}")
        else:
            device = torch.device("cpu")
            print("Using CPU device")

        # Load model with Apple Silicon optimizations
        print("\nLoading model...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # Use half precision to save memory
            device_map="auto" if device.type == "mps" else "cpu",
            low_cpu_mem_usage=True,
        )
        print("✓ Model loaded successfully")

        print("Loading processor...")
        processor = AutoProcessor.from_pretrained(model_path)
        print("✓ Processor loaded successfully")

        print("\n=== Testing ===")
        test_image_larger = Image.new("RGB", (224, 224), color="blue")

        test_messages_larger = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": test_image_larger},
                    {"type": "text", "text": "What color is this image?"},
                ],
            }
        ]

        text = processor.apply_chat_template(test_messages_larger, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(test_messages_larger)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        )

        if device.type == "mps":
            inputs = inputs.to(device)

        print("Generating larger response...")
        start_time = time.time()

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=50, do_sample=False, num_beams=1)

        generation_time = time.time() - start_time
        print(f"✓ Larger generation completed in {generation_time:.2f} seconds")

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        print(f"Larger Response: {output_text[0]}")
        print("🎉 All tests passed!")

        return True

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_cpu_only(model_path="ByteDance-Seed/UI-TARS-1.5-7B"):
    """Force CPU-only test if MPS fails"""
    print("\n=== Testing CPU-only fallback ===")

    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float32, device_map="cpu"
        )
        processor = AutoProcessor.from_pretrained(model_path)

        test_image = Image.new("RGB", (32, 32), color="green")
        test_messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": test_image}, {"type": "text", "text": "Test"}],
            }
        ]

        text = processor.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(test_messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        )

        print("Generating on CPU...")
        start_time = time.time()

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)

        generation_time = time.time() - start_time
        print(f"CPU generation completed in {generation_time:.2f} seconds")

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        print(f"CPU Response: {output_text[0]}")
        return True

    except Exception as e:
        print(f"❌ CPU test also failed: {e}")
        return False


if __name__ == "__main__":
    # Install psutil if not available
    try:
        import psutil
    except ImportError:
        print("Installing psutil...")
        import subprocess

        subprocess.check_call(["pip", "install", "psutil"])
        import psutil

    # Try MPS first, then CPU fallback
    success = test_uitars_apple_silicon()

    if not success:
        test_cpu_only()
