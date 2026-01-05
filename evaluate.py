"""
Evaluate your trained B-Free model on WildRF and Synthbuster datasets
Uses 5-crop inference and reports balanced accuracy, AUC, and per-dataset results

Usage:
    # Evaluate on WildRF (all platforms)
    python evaluate_my_model.py --dataset wildrf --checkpoint checkpoints/checkpoints_4/checkpoint_epoch_10.pth
    
    # Evaluate on Synthbuster (all generators)
    python evaluate_my_model.py --dataset synthbuster --checkpoint checkpoints/checkpoints_4/checkpoint_epoch_10.pth
    
    # Quick test with 100 images per subset
    python evaluate_my_model.py --dataset both --max_images 100
"""

import torch
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from collections import defaultdict
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

from model import BFreeModelLoRA
from utils import load_config
import torchvision.transforms as transforms


def extract_5_crops(img, crop_size=504):
    """Extract 5 crops from image (author's approach)"""
    w, h = img.size
    if w < crop_size or h < crop_size:
        scale = max(crop_size / w, crop_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
    
    w, h = img.size
    hs = max((h - crop_size) // 2, 0)
    ws = max((w - crop_size) // 2, 0)
    
    crops = []
    crops.append(img.crop((ws, hs, ws + crop_size, hs + crop_size)))  # Center
    crops.append(img.crop((0, 0, crop_size, crop_size)))  # Top-left
    crops.append(img.crop((w - crop_size, 0, w, crop_size)))  # Top-right
    crops.append(img.crop((w - crop_size, h - crop_size, w, h)))  # Bottom-right
    crops.append(img.crop((0, h - crop_size, crop_size, h)))  # Bottom-left
    
    return crops


def load_model(checkpoint_path, device='cuda'):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
    
    # Auto-detect architecture
    hidden_dim = state_dict['classifier.weight'].shape[1]
    model_map = {
        768: "facebook/dinov2-with-registers-base",
        1024: "facebook/dinov2-with-registers-large",
        1536: "facebook/dinov2-with-registers-giant"
    }
    model_name = model_map.get(hidden_dim)
    
    # Load config
    config = load_config('config.yaml')
    lora_cfg = config.get('training', {}).get('lora', {})
    
    model = BFreeModelLoRA(
        model_name=model_name,
        num_classes=2,
        lora_r=lora_cfg.get('r', 16),
        lora_alpha=lora_cfg.get('alpha', 32),
        lora_dropout=lora_cfg.get('dropout', 0.1),
        target_modules=lora_cfg.get('target_modules', ["query", "value"])
    )
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    epoch = checkpoint.get('epoch', 'N/A')
    print(f"✓ Loaded {model_name.split('-')[-1].upper()} model (Epoch: {epoch}) on {device}")
    
    return model, model_name


def predict_image(model, image_path, device, use_5crop=True):
    """Predict single image with 5-crop"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    
    with torch.no_grad():
        if use_5crop:
            crops = extract_5_crops(img, crop_size=504)
            crop_tensors = torch.stack([transform(crop) for crop in crops]).to(device)
            outputs = model(crop_tensors)
            avg_logits = outputs.mean(dim=0)
            probs = torch.softmax(avg_logits, dim=0)
        else:
            img_resized = img.resize((504, 504), Image.BILINEAR)
            img_tensor = transform(img_resized).unsqueeze(0).to(device)
            outputs = model(img_tensor)
            probs = torch.softmax(outputs[0], dim=0)
    
    return probs[1].item()  # prob_fake


def evaluate_wildrf(model, device, wildrf_dir, max_images=None, use_5crop=True):
    """Evaluate on WildRF dataset (Facebook, Reddit, Twitter)"""
    results = {}
    all_labels = []
    all_probs = []
    
    platforms = ['facebook', 'reddit', 'twitter']
    
    for platform in platforms:
        print(f"\n{'='*60}")
        print(f"Evaluating WildRF - {platform.upper()}")
        print(f"{'='*60}")
        
        # Real images (0_real)
        real_dir = Path(wildrf_dir) / 'test' / platform / '0_real'
        # Fake images (1_fake)
        fake_dir = Path(wildrf_dir) / 'test' / platform / '1_fake'
        
        if not real_dir.exists() or not fake_dir.exists():
            print(f"⚠ Skipping {platform}: directories not found")
            continue
        
        # Collect images
        real_images = list(real_dir.glob('*.*'))
        fake_images = list(fake_dir.glob('*.*'))
        
        if max_images:
            real_images = real_images[:max_images]
            fake_images = fake_images[:max_images]
        
        print(f"Real: {len(real_images)}, Fake: {len(fake_images)}")
        
        # Predict
        platform_labels = []
        platform_probs = []
        
        for img_path in tqdm(real_images, desc="Real"):
            try:
                prob = predict_image(model, img_path, device, use_5crop)
                platform_labels.append(0)
                platform_probs.append(prob)
            except Exception as e:
                print(f"Error: {img_path.name}: {e}")
        
        for img_path in tqdm(fake_images, desc="Fake"):
            try:
                prob = predict_image(model, img_path, device, use_5crop)
                platform_labels.append(1)
                platform_probs.append(prob)
            except Exception as e:
                print(f"Error: {img_path.name}: {e}")
        
        # Compute metrics
        labels = np.array(platform_labels)
        probs = np.array(platform_probs)
        preds = (probs > 0.5).astype(int)
        
        auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0
        bacc = balanced_accuracy_score(labels, preds)
        real_acc = np.mean(preds[labels == 0] == 0)
        fake_acc = np.mean(preds[labels == 1] == 1)
        
        results[platform] = {
            'auc': float(auc),
            'balanced_accuracy': float(bacc),
            'real_accuracy': float(real_acc),
            'fake_accuracy': float(fake_acc),
            'num_real': int(np.sum(labels == 0)),
            'num_fake': int(np.sum(labels == 1))
        }
        
        print(f"AUC: {auc:.4f}, bAcc: {bacc:.4f}, Real: {real_acc:.4f}, Fake: {fake_acc:.4f}")
        
        all_labels.extend(platform_labels)
        all_probs.extend(platform_probs)
    
    # Overall WildRF metrics
    if all_labels:
        labels = np.array(all_labels)
        probs = np.array(all_probs)
        preds = (probs > 0.5).astype(int)
        
        results['overall'] = {
            'auc': float(roc_auc_score(labels, probs)),
            'balanced_accuracy': float(balanced_accuracy_score(labels, preds)),
            'real_accuracy': float(np.mean(preds[labels == 0] == 0)),
            'fake_accuracy': float(np.mean(preds[labels == 1] == 1)),
            'num_real': int(np.sum(labels == 0)),
            'num_fake': int(np.sum(labels == 1))
        }
        
        print(f"\n{'='*60}")
        print("WILDRF OVERALL")
        print(f"{'='*60}")
        print(f"AUC: {results['overall']['auc']:.4f}")
        print(f"Balanced Accuracy: {results['overall']['balanced_accuracy']:.4f}")
        print(f"Real Accuracy: {results['overall']['real_accuracy']:.4f}")
        print(f"Fake Accuracy: {results['overall']['fake_accuracy']:.4f}")
    
    return results


def evaluate_synthbuster(model, device, synthbuster_dir, max_images=None, use_5crop=True):
    """Evaluate on Synthbuster dataset (9 generators)"""
    results = {}
    all_probs = []
    
    # All generators in Synthbuster
    generators = [
        'dalle2', 'dalle3', 'firefly', 
        'midjourney-v5', 'stable-diffusion-1-3', 'stable-diffusion-1-4', 'stable-diffusion-2', 
        'stable-diffusion-xl'
    ]
    
    for generator in generators:
        print(f"\n{'='*60}")
        print(f"Evaluating Synthbuster - {generator.upper()}")
        print(f"{'='*60}")
        
        gen_dir = Path(synthbuster_dir) / generator
        
        if not gen_dir.exists():
            print(f"⚠ Skipping {generator}: directory not found")
            continue
        
        # Collect images
        images = list(gen_dir.glob('*.png')) + list(gen_dir.glob('*.jpg'))
        
        if max_images:
            images = images[:max_images]
        
        print(f"Images: {len(images)}")
        
        # Predict (all are fake)
        gen_probs = []
        
        for img_path in tqdm(images, desc=generator):
            try:
                prob = predict_image(model, img_path, device, use_5crop)
                gen_probs.append(prob)
            except Exception as e:
                print(f"Error: {img_path.name}: {e}")
        
        # Metrics (all are fake, so label=1)
        probs = np.array(gen_probs)
        preds = (probs > 0.5).astype(int)
        accuracy = np.mean(preds == 1)  # Should predict fake
        
        results[generator] = {
            'accuracy': float(accuracy),
            'mean_prob_fake': float(np.mean(probs)),
            'std_prob_fake': float(np.std(probs)),
            'num_images': len(gen_probs)
        }
        
        print(f"Accuracy: {accuracy:.4f}, Mean Prob(Fake): {np.mean(probs):.4f}")
        
        all_probs.extend(gen_probs)
    
    # Overall Synthbuster metrics
    if all_probs:
        probs = np.array(all_probs)
        preds = (probs > 0.5).astype(int)
        
        results['overall'] = {
            'accuracy': float(np.mean(preds == 1)),
            'mean_prob_fake': float(np.mean(probs)),
            'std_prob_fake': float(np.std(probs)),
            'num_images': len(all_probs)
        }
        
        print(f"\n{'='*60}")
        print("SYNTHBUSTER OVERALL")
        print(f"{'='*60}")
        print(f"Accuracy: {results['overall']['accuracy']:.4f}")
        print(f"Mean Prob(Fake): {results['overall']['mean_prob_fake']:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate B-Free model')
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints/checkpoints_4/checkpoint_epoch_10.pth',
                       help='Checkpoint path')
    parser.add_argument('--dataset', type=str, choices=['wildrf', 'synthbuster', 'both'],
                       default='both', help='Which dataset to evaluate')
    parser.add_argument('--wildrf_dir', type=str, default='data/WildRF/WildRF',
                       help='WildRF dataset directory')
    parser.add_argument('--synthbuster_dir', type=str, default='data/synthbuster/synthbuster',
                       help='Synthbuster dataset directory')
    parser.add_argument('--max_images', type=int, default=None,
                       help='Max images per subset (for quick testing)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--single_crop', action='store_true',
                       help='Use single crop instead of 5-crop')
    parser.add_argument('--output', type=str, default='my_model_evaluation.json',
                       help='Output JSON file')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model, model_name = load_model(args.checkpoint, device=args.device)
    
    use_5crop = not args.single_crop
    print(f"Inference method: {'5-crop' if use_5crop else 'single-crop'}")
    
    results = {
        'checkpoint': args.checkpoint,
        'model': model_name,
        'inference_method': '5-crop' if use_5crop else 'single-crop',
        'max_images_per_subset': args.max_images
    }
    
    # Evaluate WildRF
    if args.dataset in ['wildrf', 'both']:
        print(f"\n{'#'*60}")
        print("EVALUATING WILDRF DATASET")
        print(f"{'#'*60}")
        results['wildrf'] = evaluate_wildrf(
            model, args.device, args.wildrf_dir, 
            max_images=args.max_images, use_5crop=use_5crop
        )
    
    # Evaluate Synthbuster
    if args.dataset in ['synthbuster', 'both']:
        print(f"\n{'#'*60}")
        print("EVALUATING SYNTHBUSTER DATASET")
        print(f"{'#'*60}")
        results['synthbuster'] = evaluate_synthbuster(
            model, args.device, args.synthbuster_dir,
            max_images=args.max_images, use_5crop=use_5crop
        )
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {args.output}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
