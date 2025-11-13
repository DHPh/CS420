"""
Step 4: Generate inpainted images - different category
Replaces objects with objects from different categories.
"""
import os
import json
import argparse
import random
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image, ImageDraw
import numpy as np
from diffusers import StableDiffusionInpaintPipeline
from pycocotools.coco import COCO

from utils import load_config, set_seed


def bbox_to_mask(bbox, image_size):
    """Convert bounding box to binary mask"""
    mask = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(mask)
    x, y, w, h = bbox
    draw.rectangle([x, y, x + w, y + h], fill=255)
    return mask


def get_random_category(coco, exclude_category_id=None):
    """Get a random category name, optionally excluding one"""
    cats = coco.loadCats(coco.getCatIds())
    
    if exclude_category_id:
        cats = [cat for cat in cats if cat['id'] != exclude_category_id]
    
    selected_cat = random.choice(cats)
    return selected_cat['name']


def generate_inpainted_different_category(config, args):
    """Generate inpainted images with different category replacement"""
    
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
    
    # Load bbox info
    bbox_dir = Path(data_config['bbox_dir'])
    with open(bbox_dir / 'bbox_info.json', 'r') as f:
        bbox_info = json.load(f)
    
    # Create output directories
    output_dir = Path(data_config['inpaint_diffcat_dir'])
    output_dir_origbg = Path(data_config['inpaint_diffcat_origbg_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_origbg.mkdir(parents=True, exist_ok=True)
    
    successful = 0
    failed = 0
    
    # Process images
    for img_filename, bboxes in tqdm(list(bbox_info.items()), desc="Inpainting different category"):
        try:
            # Load original image
            img_path = os.path.join(data_config['coco_train_images'], img_filename)
            image = Image.open(img_path).convert('RGB')
            
            # Resize if needed
            original_size = image.size
            if max(image.size) > data_config['image_size']:
                image.thumbnail((data_config['image_size'], data_config['image_size']), Image.LANCZOS)
                scale_factor = image.size[0] / original_size[0]
            else:
                scale_factor = 1.0
            
            # Select a random bbox from this image
            selected_bbox = random.choice(bboxes)
            original_category_id = selected_bbox['category_id']
            
            # Scale bbox if image was resized
            bbox = selected_bbox['bbox']
            if scale_factor != 1.0:
                bbox = [coord * scale_factor for coord in bbox]
            
            # Create mask from bbox
            mask = bbox_to_mask(bbox, image.size)
            
            # Get a different category
            new_category = get_random_category(coco, exclude_category_id=original_category_id)
            prompt = f"a {new_category}"
            
            # Generate inpainted image
            result = pipe(
                prompt=prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=data_config['num_inference_steps'],
                guidance_scale=data_config['guidance_scale'],
                generator=torch.Generator(device=device).manual_seed(
                    data_config['seed'] + hash(img_filename) + 1000
                )
            ).images[0]
            
            # Save full inpainted image
            output_path = output_dir / img_filename
            result.save(output_path)
            
            # Create version with original background
            mask_np = np.array(mask) / 255.0
            mask_np = mask_np[:, :, None]
            
            result_np = np.array(result)
            image_np = np.array(image)
            
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
    parser = argparse.ArgumentParser(description="Generate inpainted images (different category)")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to configuration file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    generate_inpainted_different_category(config, args)


if __name__ == "__main__":
    main()
