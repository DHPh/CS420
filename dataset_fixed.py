"""
FIXED Dataset class for B-Free training
KEY FIX: Resamples different fake images each epoch
"""
import os
import random
from pathlib import Path
from typing import List, Tuple, Optional
import json

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np


class BFreeDataset(Dataset):
    """
    Dataset for B-Free AI-generated image detection
    
    FIXED: Resamples different fake images each epoch to use all available data
    
    Combines real images from COCO with various types of synthetic images:
    - Self-conditioned reconstructions
    - Inpainted (same category)
    - Inpainted (different category)
    - Variants with original backgrounds
    """
    
    def __init__(
        self,
        real_image_dir: str,
        fake_image_dirs: List[str],
        crop_size: int = 504,
        is_train: bool = True,
        use_augmentation: bool = True,
        config: dict = None
    ):
        self.real_image_dir = Path(real_image_dir)
        self.fake_image_dirs = [Path(d) for d in fake_image_dirs]
        self.crop_size = crop_size
        self.is_train = is_train
        self.use_augmentation = use_augmentation
        self.config = config or {}
        
        # Collect all image paths
        self.real_images = self._collect_images(self.real_image_dir)
        self.fake_images = []
        for fake_dir in self.fake_image_dirs:
            self.fake_images.extend(self._collect_images(fake_dir))
        
        print(f"Dataset: {len(self.real_images)} real, {len(self.fake_images)} fake images")
        
        # FIX: Don't create samples in __init__, do it in set_epoch()
        self.current_epoch = 0
        self.samples = []
        # Create initial samples
        self._resample_balanced()
        
    def set_epoch(self, epoch: int):
        """
        Call this at the start of each epoch to resample fake images.
        This ensures all fake images are eventually seen during training.
        
        Args:
            epoch: Current epoch number
        """
        self.current_epoch = epoch
        self._resample_balanced()
        
    def _resample_balanced(self):
        """
        Resample fake images to create balanced dataset.
        Uses epoch number as seed for reproducibility.
        """
        # Set seed based on epoch for reproducibility but variation
        base_seed = self.config.get('seed', 42)
        random.seed(base_seed + self.current_epoch)
        
        samples = []
        
        # Add all real images (label 0)
        for img_path in self.real_images:
            samples.append((img_path, 0))
        
        # Sample fake images (label 1) to match real count
        num_real = len(self.real_images)
        if len(self.fake_images) >= num_real:
            # Random sample without replacement (different each epoch!)
            sampled_fake = random.sample(self.fake_images, num_real)
        else:
            # If not enough fakes, repeat some
            sampled_fake = random.choices(self.fake_images, k=num_real)
        
        for img_path in sampled_fake:
            samples.append((img_path, 1))
        
        # Shuffle
        random.shuffle(samples)
        
        self.samples = samples
        
        if self.current_epoch == 0:
            print(f"Balanced sampling: {len([s for s in samples if s[1]==0])} real, "
                  f"{len([s for s in samples if s[1]==1])} fake")
        else:
            print(f"Epoch {self.current_epoch}: Resampled {len(sampled_fake)} fake images")
        
    def _collect_images(self, directory: Path) -> List[Path]:
        """Collect all image files from directory"""
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        images = []
        for ext in extensions:
            images.extend(directory.rglob(f'*{ext}'))
        return sorted(images)
    
    def _create_balanced_samples(self) -> List[Tuple[Path, int]]:
        """
        DEPRECATED: Use set_epoch() instead
        Kept for backward compatibility
        """
        return self._resample_balanced()
    
    def __len__(self):
        return len(self.samples)
    
    def _apply_blur(self, img: Image.Image) -> Image.Image:
        """Apply Gaussian blur"""
        if not self.config.get('use_blur', False):
            return img
        
        sigma_range = self.config.get('blur_kernel_range', [0, 4])
        sigma = random.uniform(sigma_range[0], sigma_range[1])
        
        if sigma > 0:
            from PIL import ImageFilter
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        
        return img
    
    def _apply_jpeg_compression(self, img: Image.Image) -> Image.Image:
        """Apply JPEG compression artifacts"""
        if not self.config.get('use_jpeg_compression', False):
            return img
        
        quality_range = self.config.get('jpeg_quality_range', [30, 100])
        quality = random.randint(quality_range[0], quality_range[1])
        
        # Apply JPEG compression
        from io import BytesIO
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        img = Image.open(buffer)
        
        return img
    
    def _apply_resize(self, img: Image.Image) -> Image.Image:
        """Resize to crop_size"""
        if img.size != (self.crop_size, self.crop_size):
            img = img.resize((self.crop_size, self.crop_size), Image.BILINEAR)
        return img
    
    def _apply_augmentation(self, img: Image.Image) -> Image.Image:
        """Apply data augmentation pipeline"""
        if not self.use_augmentation or not self.is_train:
            return img
        
        # 1. Random horizontal flip
        if random.random() > 0.5:
            img = TF.hflip(img)
        
        # 2. Random rotation (-10 to 10 degrees)
        if random.random() > 0.7:
            angle = random.uniform(-10, 10)
            img = TF.rotate(img, angle)
        
        # 3. Color jitter
        if random.random() > 0.5:
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            saturation = random.uniform(0.8, 1.2)
            img = TF.adjust_brightness(img, brightness)
            img = TF.adjust_contrast(img, contrast)
            img = TF.adjust_saturation(img, saturation)
        
        return img
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Apply preprocessing
        img = self._apply_blur(img)
        img = self._apply_jpeg_compression(img)
        img = self._apply_resize(img)
        img = self._apply_augmentation(img)
        
        # Convert to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(img)
        
        return img_tensor, label


# For backward compatibility, keep old name
class BFreeDatasetFixed(BFreeDataset):
    """Alias for the fixed dataset"""
    pass
