## Reproduction of "A Bias-Free Training Paradigm for More General AI-generated Image Detection" (CVPR 2025) ([paper](https://arxiv.org/pdf/2412.17671))

Best model and its logs in this [link](https://drive.google.com/drive/folders/1hSYKXaqQSOAyPXlH4qnZbT82f40dkVqc?usp=sharing)

### Data Generation Pipeline

The data generation consists of 4 steps:

#### 1. Self-Conditioned Generation

Generates synthetic images using Stable Diffusion 2.1 inpainting with empty masks:

```bash
python data_generation/01_generate_self_conditioned.py --config config.yaml
```

**Output**: 51k self-conditioned images with same content as real images

#### 2. Extract Masks and Bounding Boxes

Extracts object segmentation masks and bounding boxes from COCO annotations:

```bash
python data_generation/02_extract_masks_bboxes.py --config config.yaml
```

**Output**: Mask files + metadata JSON

#### 3. Inpainting (Same Category)

Replaces objects with other objects from the same category:

```bash
python data_generation/03_generate_inpaint_samecat.py --config config.yaml
```

**Output**: 102k images (full + original background)

#### 4. Inpainting (Different Category)

Replaces objects with objects from different categories:

```bash
python data_generation/04_generate_inpaint_diffcat.py --config config.yaml
```

**Output**: 102k images (full + original background)

**Total Dataset Size**: 51k real + 309k fake images

### Training

```bash
python train.py --config config.yaml
```

### Evaluation

Evaluate trained model on test datasets:

```bash
# Evaluate on configured datasets
python evaluate.py --config config.yaml --checkpoint checkpoints/checkpoint_best.pth

# Evaluate on custom dataset
python evaluate.py \
  --config config.yaml \
  --checkpoint checkpoints/checkpoint_best.pth \
  --test_real_dir /path/to/real \
  --test_fake_dir /path/to/fake
```