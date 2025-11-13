"""
Step 3: Generate inpainted images - same category
Replaces objects with other objects from the same category.
"""
import os
import json
import argparse
import random
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image
import numpy as np
from diffusers import StableDiffusionInpaintPipeline
from pycocotools.coco import COCO

from utils import load_config, set_seed


def dilate_mask(mask, kernel_size=5):
    """Dilate mask to include some context around the object"""
    import cv2
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(np.array(mask), kernel, iterations=1)
    return Image.fromarray(dilated)


def generate_inpainted_same_category(config, args):
    """Generate inpainted images with same category replacement"""
    
    data_config = config['data_generation']
    set_seed(data_config['seed'])
    
    device = torch.device(data_config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model: {data_config['model_id']}")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        data_config['model_id'],
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    
    # Load COCO
    coco = COCO(data_config['coco_train_annotations'])
    
    # Load mask info
    mask_dir = Path(data_config['masks_dir'])
    with open(mask_dir / 'mask_info.json', 'r') as f:
        mask_info = json.load(f)
    
    # Create output directories
    output_dir = Path(data_config['inpaint_samecat_dir'])
    output_dir_origbg = Path(data_config['inpaint_samecat_origbg_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_origbg.mkdir(parents=True, exist_ok=True)
    
    successful = 0
    failed = 0
    
    # Process images
    for img_filename, masks in tqdm(list(mask_info.items()), desc="Inpainting same category"):
        try:
            # Load original image
            img_path = os.path.join(data_config['coco_train_images'], img_filename)
            image = Image.open(img_path).convert('RGB')
            
            # Resize if needed
            if max(image.size) > data_config['image_size']:
                image.thumbnail((data_config['image_size'], data_config['image_size']), Image.LANCZOS)
            
            # Select a random mask from this image
            selected_mask = random.choice(masks)
            category_name = selected_mask['category_name']
            
            # Load mask
            mask_path = mask_dir / selected_mask['mask_file']
            mask = Image.open(mask_path).convert('L')
            
            # Resize mask to match image
            if mask.size != image.size:
                mask = mask.resize(image.size, Image.NEAREST)
            
            # Dilate mask slightly
            mask = dilate_mask(mask, kernel_size=5)
            
            # Generate prompt for same category
            prompt = f"a {category_name}"
            
            # Generate inpainted image
            result = pipe(
                prompt=prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=data_config['num_inference_steps'],
                guidance_scale=data_config['guidance_scale'],
                generator=torch.Generator(device=device).manual_seed(
                    data_config['seed'] + hash(img_filename)
                )
            ).images[0]
            
            # Save full inpainted image
            output_path = output_dir / img_filename
            result.save(output_path)
            
            # Create version with original background
            # Only replace the masked region
            mask_np = np.array(mask) / 255.0
            mask_np = mask_np[:, :, None]  # Add channel dimension
            
            result_np = np.array(result)
            image_np = np.array(image)
            
            # Blend: use result where mask is 1, original where mask is 0
            blended = (result_np * mask_np + image_np * (1 - mask_np)).astype(np.uint8)
            blended_img = Image.fromarray(blended)
            
            # Save with original background
            output_path_origbg = output_dir_origbg / img_filename
            blended_img.save(output_path_origbg)
            
            successful += 1
            
        except Exception as e:
            print(f"Error processing {img_filename}: {e}")
            failed += 1
            continue
    
    print(f"\nInpainting complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory (full): {output_dir}")
    print(f"Output directory (original BG): {output_dir_origbg}")


def main():
    parser = argparse.ArgumentParser(description="Generate inpainted images (same category)")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to configuration file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    generate_inpainted_same_category(config, args)


if __name__ == "__main__":
    main()
