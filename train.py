"""
Training script for B-Free with multi-GPU support (DDP)
Configured for Vast.ai persistent execution
"""
import os
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from dataset import BFreeDataset
from model import build_model
from utils import (
    load_config, set_seed, setup_distributed, is_main_process,
    save_checkpoint, load_checkpoint, AverageMeter, reduce_dict
)


def train_one_epoch(
    model, dataloader, criterion, optimizer, scaler, epoch, config, rank, world_size
):
    """Train for one epoch"""
    model.train()
    
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    if is_main_process(rank):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    else:
        pbar = dataloader
    
    for step, (images, labels) in enumerate(pbar):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast(enabled=config['training']['mixed_precision']):
            logits = model(pixel_values=images)
            loss = criterion(logits, labels)
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient clipping
        if config['training']['gradient_clip'] > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training']['gradient_clip']
            )
        
        scaler.step(optimizer)
        scaler.update()
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        # Update meters
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc.item(), images.size(0))
        
        # Logging
        if is_main_process(rank) and step % config['training']['log_every_n_steps'] == 0:
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'acc': f'{acc_meter.avg:.4f}'
            })
    
    # Synchronize metrics across GPUs
    metrics = {
        'train_loss': torch.tensor(loss_meter.avg).cuda(),
        'train_acc': torch.tensor(acc_meter.avg).cuda()
    }
    metrics = reduce_dict(metrics, average=True)
    
    return metrics['train_loss'].item(), metrics['train_acc'].item()


@torch.no_grad()
def validate(model, dataloader, criterion, config, rank):
    """Validate the model"""
    model.eval()
    
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    if is_main_process(rank):
        pbar = tqdm(dataloader, desc="Validation")
    else:
        pbar = dataloader
    
    for images, labels in pbar:
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        
        with autocast(enabled=config['training']['mixed_precision']):
            logits = model(pixel_values=images)
            loss = criterion(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc.item(), images.size(0))
        
        if is_main_process(rank):
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'acc': f'{acc_meter.avg:.4f}'
            })
    
    # Synchronize metrics
    metrics = {
        'val_loss': torch.tensor(loss_meter.avg).cuda(),
        'val_acc': torch.tensor(acc_meter.avg).cuda()
    }
    metrics = reduce_dict(metrics, average=True)
    
    return metrics['val_loss'].item(), metrics['val_acc'].item()


def main(args):
    # Load config
    config = load_config(args.config)
    
    # Setup distributed training
    is_distributed, rank, world_size, gpu = setup_distributed()
    
    if is_main_process(rank):
        print("="*50)
        print("B-Free Training")
        print("="*50)
        print(f"Distributed: {is_distributed}")
        print(f"World size: {world_size}")
        print(f"Rank: {rank}")
    
    # Set seed
    set_seed(config['data_generation']['seed'] + rank)
    
    # Create datasets
    if is_main_process(rank):
        print("\nLoading datasets...")
    
    # Collect all fake image directories
    fake_dirs = [
        config['data_generation']['self_conditioned_dir'],
        config['data_generation']['self_conditioned_origbg_dir'],
        config['data_generation']['inpaint_samecat_dir'],
        config['data_generation']['inpaint_samecat_origbg_dir'],
        config['data_generation']['inpaint_diffcat_dir'],
        config['data_generation']['inpaint_diffcat_origbg_dir'],
    ]
    
    # Filter directories that exist
    fake_dirs = [d for d in fake_dirs if os.path.exists(d)]
    
    if is_main_process(rank):
        print(f"Using {len(fake_dirs)} fake image directories")
    
    # Create full dataset
    full_dataset = BFreeDataset(
        real_image_dir=config['data_generation']['coco_train_images'],
        fake_image_dirs=fake_dirs,
        crop_size=config['training']['crop_size'],
        is_train=True,
        use_augmentation=True,
        config=config['training']
    )
    
    # Split into train/val
    train_size = int(config['training']['train_ratio'] * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        sampler=val_sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    if is_main_process(rank):
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
    
    # Build model
    if is_main_process(rank):
        print("\nBuilding model...")
        if config['training'].get('use_lora', False):
            print("Using LoRA for parameter-efficient fine-tuning")
    
    use_lora = config['training'].get('use_lora', False)
    model = build_model(config, linear_probe=args.linear_probe, use_lora=use_lora)
    model = model.cuda()
    
    if is_distributed:
        model = DDP(model, device_ids=[gpu], find_unused_parameters=False)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    if config['training']['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['training']['optimizer']}")
    
    # Learning rate scheduler
    if config['training']['lr_scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['num_epochs']
        )
    else:
        scheduler = None
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=config['training']['mixed_precision'])
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if config['training']['resume_from'] and os.path.exists(config['training']['resume_from']):
        if is_main_process(rank):
            print(f"\nResuming from checkpoint: {config['training']['resume_from']}")
        
        start_epoch, best_val_loss = load_checkpoint(
            config['training']['resume_from'],
            model.module if is_distributed else model,
            optimizer,
            scheduler
        )
    
    # Training loop
    if is_main_process(rank):
        print("\n" + "="*50)
        print("Starting training...")
        print("="*50)
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        if is_distributed:
            train_sampler.set_epoch(epoch)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            epoch, config, rank, world_size
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, config, rank)
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Log
        if is_main_process(rank):
            print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        if is_main_process(rank):
            is_best = val_loss < best_val_loss
            best_val_loss = min(val_loss, best_val_loss)
            
            if (epoch + 1) % config['training']['save_every_n_epochs'] == 0 or is_best:
                checkpoint_dir = Path(config['training']['checkpoint_dir'])
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
                
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'model_state_dict': model.module.state_dict() if is_distributed else model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'best_val_loss': best_val_loss,
                        'config': config
                    },
                    str(checkpoint_path),
                    is_best=is_best
                )
                
                print(f"Saved checkpoint: {checkpoint_path}")
    
    if is_main_process(rank):
        print("\n" + "="*50)
        print("Training complete!")
        print("="*50)
    
    # Cleanup
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train B-Free model")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--linear_probe", action="store_true",
                       help="Use linear probing only (freeze backbone)")
    args = parser.parse_args()
    
    main(args)
