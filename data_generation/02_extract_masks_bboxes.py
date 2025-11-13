"""
Step 2: Extract masks and bounding boxes from COCO annotations
Prepares masks and bounding boxes for inpainting operations.
"""
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

from utils import load_config


def extract_masks_and_bboxes(config, args):
    """Extract object masks and bounding boxes from COCO annotations"""
    
    data_config = config['data_generation']
    
    # Load COCO annotations
    print(f"Loading COCO annotations from: {data_config['coco_train_annotations']}")
    coco = COCO(data_config['coco_train_annotations'])
    
    # Get all image IDs
    img_ids = coco.getImgIds()
    
    # Limit if specified
    if data_config['max_images'] is not None:
        img_ids = img_ids[:data_config['max_images']]
    
    # Create output directories
    mask_dir = Path(data_config['masks_dir'])
    bbox_dir = Path(data_config['bbox_dir'])
    mask_dir.mkdir(parents=True, exist_ok=True)
    bbox_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each image
    mask_info = {}
    bbox_info = {}
    
    for img_id in tqdm(img_ids, desc="Extracting masks and bboxes"):
        try:
            # Get image info
            img_info = coco.loadImgs(img_id)[0]
            img_filename = img_info['file_name']
            
            # Handle .jpg vs .png extension
            if not os.path.exists(os.path.join(data_config['coco_train_images'], img_filename)):
                # Try .png extension if .jpg doesn't exist
                img_filename = img_filename.replace('.jpg', '.png')
            
            img_width = img_info['width']
            img_height = img_info['height']
            
            # Get annotations for this image
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            
            if len(anns) == 0:
                continue
            
            # Store masks and bboxes for this image
            image_masks = []
            image_bboxes = []
            
            for ann in anns:
                # Get category
                cat_id = ann['category_id']
                cat_name = coco.loadCats(cat_id)[0]['name']
                
                # Extract mask
                if 'segmentation' in ann:
                    if isinstance(ann['segmentation'], list):
                        # Polygon format
                        rle = mask_utils.frPyObjects(ann['segmentation'], img_height, img_width)
                        mask = mask_utils.decode(rle)
                        if len(mask.shape) == 3:
                            mask = mask[:, :, 0]
                    else:
                        # RLE format
                        mask = mask_utils.decode(ann['segmentation'])
                    
                    # Save mask
                    mask_filename = f"{img_filename.split('.')[0]}_{ann['id']}.png"
                    mask_path = mask_dir / mask_filename
                    Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
                    
                    image_masks.append({
                        'mask_file': mask_filename,
                        'category_id': cat_id,
                        'category_name': cat_name,
                        'annotation_id': ann['id'],
                        'area': float(ann['area'])
                    })
                
                # Extract bbox
                if 'bbox' in ann:
                    bbox = ann['bbox']  # [x, y, width, height]
                    image_bboxes.append({
                        'bbox': bbox,
                        'category_id': cat_id,
                        'category_name': cat_name,
                        'annotation_id': ann['id'],
                        'area': float(bbox[2] * bbox[3])
                    })
            
            if image_masks:
                mask_info[img_filename] = image_masks
            if image_bboxes:
                bbox_info[img_filename] = image_bboxes
                
        except Exception as e:
            print(f"Error processing image {img_id}: {e}")
            continue
    
    # Save metadata
    with open(mask_dir / 'mask_info.json', 'w') as f:
        json.dump(mask_info, f, indent=2)
    
    with open(bbox_dir / 'bbox_info.json', 'w') as f:
        json.dump(bbox_info, f, indent=2)
    
    print(f"\nExtraction complete!")
    print(f"Images with masks: {len(mask_info)}")
    print(f"Images with bboxes: {len(bbox_info)}")
    print(f"Mask directory: {mask_dir}")
    print(f"Bbox directory: {bbox_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract masks and bounding boxes")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to configuration file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    extract_masks_and_bboxes(config, args)


if __name__ == "__main__":
    main()
