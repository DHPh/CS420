"""
Step 1: Self-conditioned image generation
Generates synthetic images using Stable Diffusion 2.1 inpainting with empty masks
to create self-conditioned reconstructions of real images.
"""
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image
import numpy as np
from diffusers import StableDiffusionInpaintPipeline
from pycocotools.coco import COCO

from utils import load_config, set_seed


def create_empty_mask(size):
    """Create an empty (all zeros) mask for self-conditioning"""
    return Image.fromarray(np.zeros(size, dtype=np.uint8))


def generate_self_conditioned_images(config, args):
    """Generate self-conditioned images from COCO dataset"""
    
    # Load configuration
    data_config = config['data_generation']
    
    # Set seed for reproducibility
    set_seed(data_config['seed'])
    
    # Setup device
    device = torch.device(data_config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Stable Diffusion inpainting model
    print(f"Loading model: {data_config['model_id']}")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        data_config['model_id'],
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    
    # Load COCO annotations
    print(f"Loading COCO annotations from: {data_config['coco_train_annotations']}")
    coco = COCO(data_config['coco_train_annotations'])
    
    # Get all image IDs
    img_ids = coco.getImgIds()
    
    # Limit number of images if specified
    if data_config['max_images'] is not None:
        img_ids = img_ids[:data_config['max_images']]
    
    print(f"Processing {len(img_ids)} images")
    
    # Create output directories
    output_dir_synthetic = Path(data_config['self_conditioned_dir'])
    output_dir_origbg = Path(data_config['self_conditioned_origbg_dir'])
    output_dir_synthetic.mkdir(parents=True, exist_ok=True)
    output_dir_origbg.mkdir(parents=True, exist_ok=True)
    
    # Process images
    successful = 0
    failed = 0
    
    for img_id in tqdm(img_ids, desc="Generating self-conditioned images"):
        try:
            # Load image info
            img_info = coco.loadImgs(img_id)[0]
            img_filename = img_info['file_name']
            
            # Handle .jpg vs .png extension (COCO annotations use .jpg but files might be .png)
            img_path = os.path.join(data_config['coco_train_images'], img_filename)
            if not os.path.exists(img_path):
                # Try .png extension if .jpg doesn't exist
                img_path_png = img_path.replace('.jpg', '.png')
                if os.path.exists(img_path_png):
                    img_path = img_path_png
                    img_filename = img_filename.replace('.jpg', '.png')
            
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Resize if needed
            if max(image.size) > data_config['image_size']:
                image.thumbnail((data_config['image_size'], data_config['image_size']), Image.LANCZOS)
            
            # Create empty mask (all zeros = no inpainting)
            mask = create_empty_mask(image.size[::-1])
            
            # Generate self-conditioned image with synthetic background
            # Using empty prompt forces the model to reconstruct based on the image itself
            result_synthetic = pipe(
                prompt="",
                image=image,
                mask_image=mask,
                num_inference_steps=data_config['num_inference_steps'],
                guidance_scale=data_config['guidance_scale'],
                generator=torch.Generator(device=device).manual_seed(data_config['seed'] + img_id)
            ).images[0]
            
            # Save synthetic background version
            output_path_synthetic = output_dir_synthetic / img_filename
            result_synthetic.save(output_path_synthetic)
            
            # Generate self-conditioned image with original background preserved
            # Use a different seed for variation
            result_origbg = pipe(
                prompt="",
                image=image,
                mask_image=mask,
                num_inference_steps=data_config['num_inference_steps'],
                guidance_scale=data_config['guidance_scale'],
                generator=torch.Generator(device=device).manual_seed(data_config['seed'] + img_id + 999999)
            ).images[0]
            
            # Save original background version
            output_path_origbg = output_dir_origbg / img_filename
            result_origbg.save(output_path_origbg)
            
            successful += 1
            
        except Exception as e:
            print(f"Error processing image {img_id}: {e}")
            failed += 1
            continue
    
    print(f"\nGeneration complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Synthetic BG output directory: {output_dir_synthetic}")
    print(f"Original BG output directory: {output_dir_origbg}")


def main():
    parser = argparse.ArgumentParser(description="Generate self-conditioned images")
    parser.add_argument("--config", type=str, default="config.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--start_idx", type=int, default=0,
                       help="Starting index for processing (for parallel runs)")
    parser.add_argument("--end_idx", type=int, default=None,
                       help="Ending index for processing (for parallel runs)")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Modify config if using custom indices
    if args.end_idx is not None:
        config['data_generation']['max_images'] = args.end_idx
    
    generate_self_conditioned_images(config, args)


if __name__ == "__main__":
    main()
