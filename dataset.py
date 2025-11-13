"""
Dataset class for B-Free training
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
        
        # Create balanced dataset
        self.samples = self._create_balanced_samples()
        
    def _collect_images(self, directory: Path) -> List[Path]:
        """Collect all image files from directory"""
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        images = []
        for ext in extensions:
            images.extend(directory.rglob(f'*{ext}'))
        return sorted(images)
    
    def _create_balanced_samples(self) -> List[Tuple[Path, int]]:
        """Create balanced list of (image_path, label) tuples"""
        samples = []
        
        # Add real images (label 0)
        for img_path in self.real_images:
            samples.append((img_path, 0))
        
        # Add fake images (label 1) - sample to match real count
        num_real = len(self.real_images)
        if len(self.fake_images) >= num_real:
            sampled_fake = random.sample(self.fake_images, num_real)
        else:
            # If not enough fakes, repeat some
            sampled_fake = random.choices(self.fake_images, k=num_real)
        
        for img_path in sampled_fake:
            samples.append((img_path, 1))
        
        # Shuffle
        random.shuffle(samples)
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def _apply_blur(self, img: Image.Image) -> Image.Image:
        """Apply Gaussian blur"""
        if not self.config.get('use_blur', False):
            return img
        
        sigma_range = self.config.get('blur_kernel_range', [0, 4])
        sigma = random.uniform(sigma_range[0], sigma_range[1])
        
        if sigma > 0:
            # Use ImageFilter.GaussianBlur for compatibility with older Pillow versions
            from PIL import ImageFilter
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        
        return img
    
    def _apply_jpeg(self, img: Image.Image) -> Image.Image:
        """Apply JPEG compression"""
        if not self.config.get('use_jpeg', False):
            return img
        
        quality_range = self.config.get('jpeg_quality_range', [60, 100])
        quality = random.randint(quality_range[0], quality_range[1])
        
        # Simulate JPEG compression
        from io import BytesIO
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        img = Image.open(buffer)
        
        return img
    
    def _apply_resize(self, img: Image.Image) -> Image.Image:
        """Apply random resize"""
        if not self.config.get('use_resize', False):
            return img
        
        scale_range = self.config.get('resize_scale_range', [0.5, 1.5])
        scale = random.uniform(scale_range[0], scale_range[1])
        
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size, Image.BILINEAR)
        
        return img
    
    def _apply_noise(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise"""
        if not self.config.get('use_noise', False):
            return img_tensor
        
        if random.random() < 0.5:
            noise = torch.randn_like(img_tensor) * 0.02
            img_tensor = torch.clamp(img_tensor + noise, 0, 1)
        
        return img_tensor
    
    def _apply_cutout(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Apply random cutout"""
        if not self.config.get('use_cutout', False):
            return img_tensor
        
        if random.random() < 0.3:
            _, h, w = img_tensor.shape
            mask_size = random.randint(h // 8, h // 4)
            
            y = random.randint(0, h - mask_size)
            x = random.randint(0, w - mask_size)
            
            img_tensor[:, y:y+mask_size, x:x+mask_size] = 0
        
        return img_tensor
    
    def _apply_color_jitter(self, img: Image.Image) -> Image.Image:
        """Apply color jittering"""
        if not self.config.get('use_jitter', False):
            return img
        
        if random.random() < 0.5:
            jitter = transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            )
            img = jitter(img)
        
        return img
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Apply augmentations (before crop)
        if self.is_train and self.use_augmentation:
            img = self._apply_color_jitter(img)
            img = self._apply_resize(img)
            img = self._apply_blur(img)
            img = self._apply_jpeg(img)
        
        # Random crop or center crop
        if self.is_train:
            # Random crop
            if min(img.size) < self.crop_size:
                # Resize if too small
                img = TF.resize(img, self.crop_size)
            
            i, j, h, w = transforms.RandomCrop.get_params(
                img, output_size=(self.crop_size, self.crop_size)
            )
            img = TF.crop(img, i, j, h, w)
            
            # Random horizontal flip
            if random.random() < 0.5:
                img = TF.hflip(img)
        else:
            # Center crop
            if min(img.size) < self.crop_size:
                img = TF.resize(img, self.crop_size)
            img = TF.center_crop(img, self.crop_size)
        
        # Convert to tensor
        img_tensor = TF.to_tensor(img)
        
        # Apply tensor-based augmentations
        if self.is_train and self.use_augmentation:
            img_tensor = self._apply_noise(img_tensor)
            img_tensor = self._apply_cutout(img_tensor)
        
        # Normalize (ImageNet stats - used by DINOv2)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        img_tensor = normalize(img_tensor)
        
        return img_tensor, label
