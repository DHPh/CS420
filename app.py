"""
Streamlit App for B-Free AI-Generated Image Detection
Upload images to detect if they're real or AI-generated
Supports both your trained model and the author's original model
"""
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path
import sys
import os
import time
import matplotlib.pyplot as plt

# Import your model
from model import BFreeModelLoRA
from utils import load_config
import torchvision.transforms as transforms

# Conditionally import author's code when needed
def get_author_imports():
    """Lazy import author's code to avoid conflicts"""
    import sys
    import os
    import importlib.util
    
    # Add author's code path
    author_path = os.path.join(os.path.dirname(__file__), 'author_code')
    if author_path not in sys.path:
        sys.path.insert(0, author_path)
    
    # Import networks module
    from networks import get_network, load_weights
    
    # Import normalization.py directly using importlib
    norm_path = os.path.join(author_path, 'utils', 'normalization.py')
    spec = importlib.util.spec_from_file_location("author_normalization", norm_path)
    norm_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(norm_module)
    
    get_list_norm = norm_module.get_list_norm
    
    from torchvision.transforms import Compose
    
    return get_network, load_weights, get_list_norm, Compose


# Page config
st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model(model_type="your_model"):
    """
    Load model - either your trained model or author's original
    
    Args:
        model_type: "your_model" or "author_model"
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_type == "author_model":
        # Load author's original model
        try:
            # Import author's code
            get_network, load_weights, get_list_norm, Compose = get_author_imports()
            
            model_path = "checkpoints/author_weight/model_epoch_best.pth"
            arch = "timm_c5i504_vit_base_patch14_reg4_dinov2.lvd142m"
            
            model = get_network(arch, pretrained=False)
            model = load_weights(model, model_path)
            model = model.to(device).eval()
            
            return model, device, "Author's Base (5-crop built-in)", None, "author"
            
        except Exception as e:
            st.error(f"❌ Error loading author's model: {str(e)}")
            st.info("Make sure author's checkpoint exists at: checkpoints/author_weight/model_epoch_best.pth")
            st.stop()
    
    else:  # your_model
        checkpoint_path = "checkpoints/checkpoints_4/checkpoint_epoch_6.pth"
        
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
            
            # Auto-detect architecture
            hidden_dim = state_dict['classifier.weight'].shape[1]
            model_map = {
                768: ("facebook/dinov2-with-registers-base", "Base"),
                1024: ("facebook/dinov2-with-registers-large", "Large"),
                1536: ("facebook/dinov2-with-registers-giant", "Giant")
            }
            model_name, model_size = model_map[hidden_dim]
            
            # Load config for LoRA settings
            config = load_config('config.yaml')
            lora_cfg = config.get('training', {}).get('lora', {})
            
            # Create model
            model = BFreeModelLoRA(
                model_name=model_name,
                num_classes=2,
                lora_r=lora_cfg.get('r', 16),
                lora_alpha=lora_cfg.get('alpha', 32),
                lora_dropout=lora_cfg.get('dropout', 0.1),
                target_modules=lora_cfg.get('target_modules', ["query", "value"])
            )
            
            # Load weights
            model.load_state_dict(state_dict)
            model.eval()
            model = model.to(device)
            
            epoch = checkpoint.get('epoch', 8)
            
            return model, device, model_size, epoch, "yours"
            
        except FileNotFoundError:
            st.error(f"❌ Checkpoint not found: {checkpoint_path}")
            st.info("Please ensure the checkpoint file exists at the specified path.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            st.stop()


def extract_5_crops(img, crop_size=504):
    """
    Extract 5 crops from image using optimal resize
    Returns list of PIL Images: [center, top-left, top-right, bottom-left, bottom-right]
    """
    # Optimal resize: 1.15x crop_size to allow proper corner extraction
    target_size = int(crop_size * 1.15)  # 579 for 504
    
    w, h = img.size
    # Resize if image is smaller than target
    if w < target_size or h < target_size:
        scale = max(target_size / w, target_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
    
    w, h = img.size
    
    # Calculate crop positions
    hs = max((h - crop_size) // 2, 0)
    ws = max((w - crop_size) // 2, 0)
    
    crops = []
    # Center crop
    crops.append(img.crop((ws, hs, ws + crop_size, hs + crop_size)))
    # Top-left
    crops.append(img.crop((0, 0, crop_size, crop_size)))
    # Top-right
    crops.append(img.crop((w - crop_size, 0, w, crop_size)))
    # Bottom-right
    crops.append(img.crop((w - crop_size, h - crop_size, w, h)))
    # Bottom-left
    crops.append(img.crop((0, h - crop_size, crop_size, h)))
    
    return crops


def extract_10_crops(img, crop_size=504):
    """
    Extract 10 crops from image for more comprehensive coverage
    Returns list of PIL Images:
    - 4 corners + center (like 5-crop)
    - 4 edge midpoints (top-center, bottom-center, left-center, right-center)
    - 1 slightly zoomed center crop
    """
    # Optimal resize: 1.2x crop_size for better edge coverage
    target_size = int(crop_size * 1.2)  # 605 for 504
    
    w, h = img.size
    # Resize if image is smaller than target
    if w < target_size or h < target_size:
        scale = max(target_size / w, target_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
    
    w, h = img.size
    
    crops = []
    
    # 1. Center crop
    hs = max((h - crop_size) // 2, 0)
    ws = max((w - crop_size) // 2, 0)
    crops.append(img.crop((ws, hs, ws + crop_size, hs + crop_size)))
    
    # 2-5. Four corners
    crops.append(img.crop((0, 0, crop_size, crop_size)))  # Top-left
    crops.append(img.crop((w - crop_size, 0, w, crop_size)))  # Top-right
    crops.append(img.crop((w - crop_size, h - crop_size, w, h)))  # Bottom-right
    crops.append(img.crop((0, h - crop_size, crop_size, h)))  # Bottom-left
    
    # 6-9. Four edge midpoints
    # Top-center
    ws_top = max((w - crop_size) // 2, 0)
    crops.append(img.crop((ws_top, 0, ws_top + crop_size, crop_size)))
    
    # Bottom-center
    ws_bottom = max((w - crop_size) // 2, 0)
    crops.append(img.crop((ws_bottom, h - crop_size, ws_bottom + crop_size, h)))
    
    # Left-center
    hs_left = max((h - crop_size) // 2, 0)
    crops.append(img.crop((0, hs_left, crop_size, hs_left + crop_size)))
    
    # Right-center
    hs_right = max((h - crop_size) // 2, 0)
    crops.append(img.crop((w - crop_size, hs_right, w, hs_right + crop_size)))
    
    # 10. Slightly zoomed center crop (90% scale for different perspective)
    zoom_size = int(crop_size * 0.9)  # 454 for 504
    zoomed_img = img.resize((w, h), Image.BILINEAR)
    hs_zoom = max((h - zoom_size) // 2, 0)
    ws_zoom = max((w - zoom_size) // 2, 0)
    zoomed_crop = zoomed_img.crop((ws_zoom, hs_zoom, ws_zoom + zoom_size, hs_zoom + zoom_size))
    zoomed_crop = zoomed_crop.resize((crop_size, crop_size), Image.BILINEAR)
    crops.append(zoomed_crop)
    
    return crops


threshold = 0.5  # Fake probability threshold


def extract_attention_heatmap(model, image, device, model_source="yours"):
    """
    Extract attention-based feature heatmap from DINOv2
    Shows which regions the model focuses on for detection
    
    Args:
        model: Loaded model
        image: PIL Image
        device: torch device
        model_source: "yours" or "author"
    
    Returns:
        heatmap_overlay: PIL Image with heatmap overlay
        attention_map: numpy array of raw attention values
    """
    if model_source == "author":
        # Author's model doesn't expose attention easily
        return None, None
    
    # Standard transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Prepare image - resize to 504x504
    img_resized = image.resize((504, 504), Image.BILINEAR)
    img_tensor = transform(img_resized).unsqueeze(0).to(device)
    
    try:
        # Get the actual backbone (unwrap LoRA if present)
        if hasattr(model, 'backbone'):
            backbone = model.backbone
            # For LoRA models, get the base model
            if hasattr(backbone, 'base_model'):
                base_backbone = backbone.base_model.model
            else:
                base_backbone = backbone
        else:
            return None, None
        
        # Set attention implementation to 'eager' to enable attention output
        # This is required for newer transformers versions with SDPA
        if hasattr(base_backbone.config, '_attn_implementation'):
            base_backbone.config._attn_implementation = 'eager'
        
        # Forward pass with attention output
        with torch.no_grad():
            # Use the wrapped backbone (with LoRA) but request attention output
            outputs = backbone(pixel_values=img_tensor, output_attentions=True)
            
            # Get attention from last layer
            # Shape: (batch, num_heads, seq_len, seq_len)
            # For DINOv2: seq_len = 1 (CLS) + 4 (registers) + num_patches
            if outputs.attentions is None or len(outputs.attentions) == 0:
                return None, None
            
            attentions = outputs.attentions[-1]  # Last layer
        
        # Calculate number of patches based on actual attention shape
        seq_len = attentions.shape[-1]
        # seq_len = 1 (CLS) + 4 (registers) + patches
        num_patches = seq_len - 5  # Subtract CLS and 4 registers
        patch_grid_size = int(np.sqrt(num_patches))  # Should be 36 for 504x504 input
        
        # Validate
        if patch_grid_size * patch_grid_size != num_patches:
            # Fallback: try without registers (some models may differ)
            num_patches = seq_len - 1  # Just CLS
            patch_grid_size = int(np.sqrt(num_patches))
            register_offset = 1
        else:
            register_offset = 5  # CLS + 4 registers
        
        # Extract CLS token's attention to patches
        # attentions shape: (1, num_heads, seq_len, seq_len)
        num_heads = attentions.shape[1]
        
        # Get attention from CLS token (index 0) to patch tokens
        cls_attention = attentions[0, :, 0, register_offset:]  # Shape: (num_heads, num_patches)
        
        # Average across all attention heads
        avg_attention = cls_attention.mean(dim=0)  # Shape: (num_patches,)
        
        # Reshape to grid
        attention_map = avg_attention.reshape(patch_grid_size, patch_grid_size).cpu().numpy()
        
        # Normalize to 0-1 range
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
        
        # Create heatmap overlay
        heatmap_overlay = create_heatmap_overlay(img_resized, attention_map)
        
        return heatmap_overlay, attention_map
        
    except Exception as e:
        print(f"Error extracting attention: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def create_heatmap_overlay(original_image, attention_map, alpha=0.5, colormap='jet'):
    """
    Create heatmap overlay on original image
    
    Args:
        original_image: PIL Image (504x504)
        attention_map: numpy array (36x36)
        alpha: overlay transparency
        colormap: matplotlib colormap name
    
    Returns:
        PIL Image with heatmap overlay
    """
    # Resize attention map to image size
    attention_resized = np.array(Image.fromarray(
        (attention_map * 255).astype(np.uint8)
    ).resize(original_image.size, Image.BILINEAR)) / 255.0
    
    # Apply colormap (use new API)
    cmap = plt.colormaps[colormap]
    heatmap_colored = cmap(attention_resized)[:, :, :3]  # RGB only
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # Convert original to numpy
    original_np = np.array(original_image)
    
    # Blend images
    blended = (alpha * heatmap_colored + (1 - alpha) * original_np).astype(np.uint8)
    
    return Image.fromarray(blended)


def predict_image(model, image, device, num_crops=5, model_source="yours"):
    """
    Predict if image is real or fake
    
    Args:
        model: Loaded model
        image: PIL Image
        device: torch device
        num_crops: Number of crops to use (1, 5, or 10) - ignored for author's model
        model_source: "yours" or "author"
    
    Returns:
        prob_real, prob_fake, prediction, confidence, inference_time
    """
    start_time = time.time()
    
    # Author's model uses its own normalization and built-in 5-crop
    if model_source == "author":
        # Import normalization function
        _, _, get_list_norm, Compose = get_author_imports()
        transform = Compose(get_list_norm('resnet'))  # Author uses ResNet normalization
        
        with torch.no_grad():
            img_tensor = transform(image).unsqueeze(0).to(device)
            out = model(img_tensor)  # Built-in 5-crop averaging
            
            # Author's model outputs 1 logit (fake score)
            if out.shape[1] == 1:
                logit_fake = out[0, 0].item()
                logit_real = -logit_fake
            # Or 2 logits
            elif out.shape[1] == 2:
                logit_real = out[0, 0].item()
                logit_fake = out[0, 1].item()
            else:
                raise ValueError(f"Unexpected output shape: {out.shape}")
            
            # Convert to probabilities
            probs = torch.softmax(torch.tensor([logit_real, logit_fake]), dim=0)
            prob_real = probs[0].item()
            prob_fake = probs[1].item()
            
            prediction = "AI-GENERATED" if prob_fake > threshold else "REAL"
            confidence = max(prob_real, prob_fake)
            
            inference_time = time.time() - start_time
            return prob_real, prob_fake, prediction, confidence, inference_time
    
    # Your model - standard inference
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    with torch.no_grad():
        if num_crops == 10:
            # Extract 10 crops for maximum coverage
            crops = extract_10_crops(image, crop_size=504)
            crop_tensors = torch.stack([transform(crop) for crop in crops]).to(device)
            
            # Forward pass
            outputs = model(crop_tensors)  # [10, 2]
            
            # Average logits BEFORE softmax (critical!)
            avg_logits = outputs.mean(dim=0)  # [2]
            probs = torch.softmax(avg_logits, dim=0)
            
        elif num_crops == 5:
            # Extract 5 crops (standard)
            crops = extract_5_crops(image, crop_size=504)
            crop_tensors = torch.stack([transform(crop) for crop in crops]).to(device)
            
            # Forward pass
            outputs = model(crop_tensors)  # [5, 2]
            
            # Average logits BEFORE softmax (critical!)
            avg_logits = outputs.mean(dim=0)  # [2]
            probs = torch.softmax(avg_logits, dim=0)
            
        else:  # num_crops == 1
            # Single crop
            img_resized = image.resize((504, 504), Image.BILINEAR)
            img_tensor = transform(img_resized).unsqueeze(0).to(device)
            
            outputs = model(img_tensor)
            probs = torch.softmax(outputs[0], dim=0)
    
    prob_real = probs[0].item()
    prob_fake = probs[1].item()
    prediction = "AI-GENERATED" if prob_fake > threshold else "REAL"
    # confidence = max(prob_real, prob_fake)
    confidence = prob_fake if prediction == "AI-GENERATED" else prob_real
    
    inference_time = time.time() - start_time
    return prob_real, prob_fake, prediction, confidence, inference_time


# Main app
def main():
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # Model selection
        model_type = st.radio(
            "Select Model",
            options=["Your Trained Model", "Author's Original Model"],
            index=0,
            help=(
                "**Your Model**: DINOv2-Large with LoRA (Epoch 8, 92.85% bAcc)\n\n"
                "**Author's Model**: DINOv2-Base original implementation (Paper results)"
            )
        )
        
        model_map = {
            "Your Trained Model": "your_model",
            "Author's Original Model": "author_model"
        }
        selected_model = model_map[model_type]
        
        st.markdown("---")
        
        # Crop mode (only for your model)
        if selected_model == "your_model":
            crop_mode = st.radio(
                "Inference Method",
                # options=["Single Crop", "5-Crop", "10-Crop"],
                options=["Single Crop", "5-Crop"],
                index=1,  # Default to 5-crop
                help=(
                    "**Single Crop**: Fastest, baseline accuracy\n\n"
                    "**5-Crop**: Balanced speed/accuracy (+9% improvement)\n\n"
                    # "**10-Crop**: Best accuracy, slower (~2× time of 5-crop)"
                )
            )
            
            # Map selection to num_crops
            crop_map = {
                "Single Crop": 1,
                "5-Crop": 5,
                # "10-Crop": 10
            }
            num_crops = crop_map[crop_mode]
            
            # Heatmap toggle
            st.markdown("---")
            show_heatmap = st.checkbox(
                "🔥 Show Attention Heatmap",
                value=False,
                help="Visualize which regions the model focuses on to make its decision"
            )
            
            if show_heatmap:
                heatmap_alpha = st.slider(
                    "Heatmap Opacity",
                    min_value=0.1,
                    max_value=0.9,
                    value=0.5,
                    step=0.1,
                    help="Adjust how visible the heatmap overlay is"
                )
                
                heatmap_colormap = st.selectbox(
                    "Colormap",
                    options=["jet", "hot", "plasma", "viridis", "inferno"],
                    index=0,
                    help="Color scheme for the heatmap"
                )
        else:
            st.info("ℹ️ Author's model has built-in 5-crop inference (cannot be changed)")
            num_crops = 5
            crop_mode = "5-Crop (Built-in)"
            show_heatmap = False
            heatmap_alpha = 0.5
            heatmap_colormap = "jet"
        
        # st.markdown("---")
        
        # st.markdown("### 📊 Model Info")
        # st.info(
        #     "**Best Model: Epoch 10**\n\n"
        #     "- Balanced Accuracy: **97%**\n"
        #     "- Architecture: DINOv2-Large\n"
        #     "- Parameters: 1.02% trainable (LoRA)\n"
        #     "- Threshold: 0.5 (normal)"
        # )
        
        # st.markdown("---")
        
        # st.markdown("### ℹ️ About")
        # st.markdown(
        #     "This detector uses a DINOv2-based model trained with LoRA "
        #     "to identify AI-generated images. It achieves 97% balanced accuracy "
        #     "on the WildRF benchmark, matching state-of-the-art performance "
        #     "with only 1% trainable parameters."
        # )
        
        # st.markdown("**Paper:** B-Free (CVPR 2025)")
    
    # Main content
    st.title("AI-Generated Image Detector")
    st.markdown("Upload an image to detect if it's real or AI-generated")
    
    # Load model
    with st.spinner("Loading model..."):
        model, device, model_size, epoch, model_source = load_model(selected_model)
    
    if model_source == "author":
        st.success(f"✅ Model loaded: {model_size} on {device}")
    else:
        # st.success(f"✅ Model loaded: DINOv2-{model_size} (Epoch {epoch}) on {device}")
        st.success(f"✅ Model loaded on {device}")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['png', 'jpg', 'jpeg', 'bmp', 'jfif'],
        help="Upload a PNG, JPG, or JPEG image"
    )
    
    if uploaded_file is not None:
        # Display image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Uploaded Image")
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, use_container_width=True)
            
            # Image info
            st.caption(f"Size: {image.size[0]} x {image.size[1]} pixels")
        
        with col2:
            st.subheader("🎯 Detection Results")
            
            # Predict
            with st.spinner("Analyzing image..."):
                prob_real, prob_fake, prediction, confidence, inference_time = predict_image(
                    model, image, device, num_crops=num_crops, model_source=model_source
                )
            
            # Display prediction
            if prediction == "REAL":
                st.success(f"### ✅ {prediction}")
                col_conf, col_time = st.columns(2)
                with col_conf:
                    st.metric("Confidence", f"{confidence*100:.1f}%")
                # with col_time:
                #     st.metric("Inference Time", f"{inference_time:.2f}s")
            else:
                st.error(f"### 🤖 {prediction}")
                col_conf, col_time = st.columns(2)
                with col_conf:
                    st.metric("Confidence", f"{confidence*100:.1f}%")
                # with col_time:
                #     st.metric("Inference Time", f"{inference_time:.2f}s")
            
            # Probability bars
            # st.markdown("---")
            # st.markdown("**Probability Breakdown:**")
            
            # # Real probability
            # st.markdown(f"**Real:** {prob_real*100:.2f}%")
            # st.progress(prob_real)
            
            # # Fake probability
            # st.markdown(f"**AI-Generated:** {prob_fake*100:.2f}%")
            # st.progress(prob_fake)
            
            # Additional info
            st.markdown("---")
            st.caption(
                f"Method: {crop_mode}\n\n"
                f"Decision Threshold: {threshold}"
            )
        
        # Heatmap visualization section (below the main columns)
        if show_heatmap and model_source == "yours":
            st.markdown("---")
            st.subheader("🔥 Attention Heatmap")
            st.markdown(
                "This heatmap shows which regions the model focuses on when detecting AI artifacts. "
                "**Warm colors (red/yellow)** indicate high attention areas."
            )
            
            with st.spinner("Generating attention heatmap..."):
                heatmap_overlay, attention_map = extract_attention_heatmap(
                    model, image, device, model_source=model_source
                )
            
            if heatmap_overlay is not None:
                # Display heatmap with custom settings
                heatmap_overlay_custom = create_heatmap_overlay(
                    image.resize((504, 504), Image.BILINEAR),
                    attention_map,
                    alpha=heatmap_alpha,
                    colormap=heatmap_colormap
                )
                
                col_orig, col_heat, col_raw = st.columns(3)
                
                with col_orig:
                    st.markdown("**Original (Resized)**")
                    st.image(image.resize((504, 504), Image.BILINEAR), use_container_width=True)
                
                with col_heat:
                    st.markdown("**With Attention Overlay**")
                    st.image(heatmap_overlay_custom, use_container_width=True)
                
                with col_raw:
                    st.markdown("**Raw Attention Map**")
                    # Create raw heatmap visualization
                    fig, ax = plt.subplots(figsize=(5, 5))
                    im = ax.imshow(attention_map, cmap=heatmap_colormap)
                    ax.axis('off')
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    st.pyplot(fig)
                    plt.close()
                
                # Interpretation guide
                with st.expander("📖 How to Interpret the Heatmap"):
                    st.markdown("""
                    **Understanding the Attention Heatmap:**
                    
                    The heatmap visualizes which 14×14 pixel patches the DINOv2 model's 
                    CLS token attends to when making its real/fake decision.
                    
                    - **Warm colors (red/yellow)**: High attention - model focuses here for detection
                    - **Cool colors (blue/purple)**: Low attention - less important for decision
                    
                    **For AI-Generated Images:**
                    - Look for patterns in high-attention areas (repeating textures, edges)
                    - Model may focus on areas with typical AI artifacts:
                      - Unnatural skin textures
                      - Inconsistent lighting/shadows
                      - Repetitive patterns
                      - Edges between objects and backgrounds
                    
                    **For Real Images:**
                    - Attention is typically more distributed
                    - May focus on distinctive natural features
                    
                    **Note:** This is the attention from the last transformer layer averaged 
                    across all attention heads. It shows what the model "looks at" but not 
                    necessarily what specific artifacts it detects.
                    """)
            else:
                st.warning("⚠️ Could not generate heatmap for this model configuration.")
    
    # Instructions
    if uploaded_file is None:
        st.info("👆 Upload an image to get started!")
        
        with st.expander("📖 How to use"):
            st.markdown("""
            1. **Choose inference method** in the sidebar:
               - **Single Crop**: Fast baseline (~0.5s)
               - **5-Crop**: Balanced accuracy/speed (~1s) ⭐ Recommended
               - **10-Crop**: Maximum accuracy (~2s)
            2. **Upload an image** using the file uploader above
            3. **Wait for analysis** (time depends on crop method)
            4. **Review the results**:
               - ✅ **REAL**: Natural photograph
               - 🤖 **AI-GENERATED**: Created by AI (DALL-E, Stable Diffusion, etc.)
            5. **Check confidence**: Higher confidence = more certain prediction
            
            **Tips:**
            - 5-crop is the sweet spot for most cases
            - 10-crop gives slightly better accuracy for challenging images
            - Works on images from social media (Facebook, Twitter, Reddit)
            - Supports various AI generators (DALL-E, Midjourney, Stable Diffusion)
            """)
        
        with st.expander("🧪 Model Performance"):
            st.markdown("""
            **Benchmark Results (WildRF Twitter):**
            - Balanced Accuracy: **97.0%**
            - Real Image Accuracy: **99%**
            - AI-Generated Accuracy: **95%**
            
            **Comparison to Paper:**
            - Paper (full model): 86-97% bAcc range
            - This model (LoRA): **97.0% bAcc** (at the top!)
            - Trainable parameters: **1.02%** (highly efficient)
            
            **Supported AI Generators:**
            - DALL-E 2/3
            - Stable Diffusion (all versions)
            - Midjourney
            - And more...
            """)


if __name__ == "__main__":
    main()
