# Keratitis Detection System

This system is designed for the detection and classification of Keratitis conditions, specifically **Edema, Scar, Infection, and Normal** cases. It utilizes a Dual-Branch Deep Learning model (EfficientNet-B0) with Gated Attention Multiple Instance Learning (MIL).

## Project Structure

```text
Keratitis_Detection_System/
├── Limbus_Crop_Segmentation_System/      # Dependency: Segmentation model & utilities
├── training_results/                     # Trained models (checkpoints/best.pth)
├── 01_precompute_dataset.py              # Step 1: Masking & Polar Tiling (Precompute)
├── 02_train_model.py                     # Step 2: Training the MIL Classifier
├── 03_inference_batch.py                 # Step 3: Batch processing of new images
├── 04_inference_ui.py                    # Step 4: Real-time Gradio User Interface
└── requirements.txt                      # Project Dependencies
```

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare the Segmenter**:
   Ensure `Limbus_Crop_Segmentation_System/model_limbus_crop_unetpp_weighted.pth` exists.

## Workflow

### Step 1: Precompute Dataset
Generate filtered and tiled patches from your input images to accelerate training.
- Configure `DATA_ROOT` in `01_precompute_dataset.py`.
- Run: `python 01_precompute_dataset.py`

### Step 2: Train Model
Train the classification model using the precomputed cache.
- Configure `CACHE_ROOT` and `OUT_DIR` in `02_train_model.py`.
- Run: `python 02_train_model.py`

### Step 3: Batch Inference
Run predictions on a folder of images and sort them.
- Configure `INPUT_DIR` and `CLS_CKPT` in `03_inference_batch.py`.
- Run: `python 03_inference_batch.py`

### Step 4: Real-time UI
Launch the Gradio interface for interactive testing.
- Ensure `CLS_CKPT` points to your trained `best.pth`.
- Run: `python 04_inference_ui.py`
