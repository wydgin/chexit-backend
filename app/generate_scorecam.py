import os

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('Agg') # This tells Matplotlib to run without a GUI
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path("/Users/gingalfo/199/")
TRAINING_DIR = BASE_DIR / "Training"
MONTGOMERY_CXR_DIR = TRAINING_DIR / "montgomery_cxr"
SHENZHEN_CXR_DIR = TRAINING_DIR / "shenzhen_cxr"

OUTPUT_DIR = BASE_DIR / "efficientnet_tb_output"
PREDICTIONS_CSV = OUTPUT_DIR / "all_predictions.csv"
WEIGHTS_DIR = OUTPUT_DIR / "weights"
VISUALIZATION_DIR = OUTPUT_DIR / "scorecam_visualizations"

IMG_SIZE = 260
TEST_MODE_LIMIT = None
TEST_ONLY_TB_POSITIVE = False

# ---------------------------------------------------------------------------
# Model Builder (exactly as in training)
# ---------------------------------------------------------------------------
def build_efficientnet_model():
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
    ])
    base_model = tf.keras.applications.EfficientNetB2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights=None
    )
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        data_augmentation,
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.6),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    return model

# ---------------------------------------------------------------------------
# Custom ScoreCAM Implementation (graph‑safe)
# ---------------------------------------------------------------------------
class ScoreCAM:
    def __init__(self, model, target_layer_name='top_conv'):
        self.model = model
        
    #     # 1. Correct the index: EfficientNetB2 is the second layer (index 1)
    #     # index 0 is augmentation, index 1 is EfficientNet, index 2 is Pooling...
    #     self.base_model = model.layers[1]
        
    #     # 2. Get the target layer from WITHIN the EfficientNet model
    #     try:
    #         self.target_layer = self.base_model.get_layer(target_layer_name)
    #     except ValueError:
    #         # Fallback/Debug: Print available layers if name is wrong
    #         available_layers = [l.name for l in self.base_model.layers]
    #         print(f"Error: Layer {target_layer_name} not found. Available: {available_layers[-5:]}")
    #         raise

    #     # 3. Build the activation model using the base_model's internal graph
    #     # This prevents the "Graph Disconnected" error.
    #     self.activation_model = tf.keras.Model(
    #         inputs=self.base_model.input,
    #         outputs=[self.target_layer.output, self.base_model.output]
    #     )
    
    # def __call__(self, score_fn, x, batch_size=32):
    #     # Since 'x' is already preprocessed, we pass it to the base_model.
    #     # If your 'x' hasn't gone through the augmentation layer yet, 
    #     # you can pass it through model.layers[0](x) first, but for 
    #     # inference/ScoreCAM, it's usually better to skip augmentation.
        
    #     # Get activations from the base model
    #     activations, base_output = self.activation_model(x, training=False)
        
    #     # ... (rest of your ScoreCAM logic remains the same)

    #     # 2. Resize activations to input size
    #     input_shape = tf.shape(x)[1:3]
    #     act_resized = tf.image.resize(activations, input_shape)
    #     act_resized = act_resized[0]  # (H, W, C)
        
    #     num_channels = act_resized.shape[-1]
        
    #     # 3. Score each channel
    #     scores = []
        
    #     for c in tqdm(range(num_channels), desc="ScoreCAM channels", leave=False):
    #         channel_map = act_resized[..., c]
    #         channel_map = (channel_map - tf.reduce_min(channel_map)) / (
    #             tf.reduce_max(channel_map) - tf.reduce_min(channel_map) + 1e-10)
    #         masked_input = x * tf.expand_dims(channel_map, axis=-1)
    #         output = self.model(masked_input, training=False)
    #         score = score_fn(output).numpy().item()
    #         scores.append(score)
        
    #     scores = np.array(scores)
    #     scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        
    #     # 4. Weighted sum
    #     cam = np.zeros(input_shape, dtype=np.float32)
    #     for c, weight in enumerate(scores):
    #         channel_map = act_resized[..., c].numpy()
    #         channel_map = (channel_map - channel_map.min()) / (channel_map.max() - channel_map.min() + 1e-10)
    #         cam += weight * channel_map
        
    #     cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-10)
    #     return cam

class ScoreCAM:
    def __init__(self, model, target_layer_name='top_conv'):
        self.model = model
        self.base_model = model.layers[1]
        
        try:
            self.target_layer = self.base_model.get_layer(target_layer_name)
        except ValueError:
            available_layers = [l.name for l in self.base_model.layers]
            raise ValueError(f"Layer {target_layer_name} not found. Try: {available_layers[-5:]}")

        self.activation_model = tf.keras.Model(
            inputs=self.base_model.input,
            outputs=[self.target_layer.output, self.base_model.output]
        )
    
    def __call__(self, score_fn, x, batch_size=16, max_channels=256):
        # 1. Get activations
        activations, _ = self.activation_model(x, training=False)
        
        # 2. Resize and Normalize
        input_shape = tf.shape(x)[1:3]
        act_resized = tf.image.resize(activations, input_shape)[0] # (H, W, C)
        
        # Normalize channels
        mins = tf.reduce_min(act_resized, axis=[0, 1])
        maxs = tf.reduce_max(act_resized, axis=[0, 1])
        act_normalized = (act_resized - mins) / (maxs - mins + 1e-10)
        
        # 3. Speed Trick: Select Top-N channels by variance
        # Most channels in EfficientNet are sparse/empty; we only need the active ones.
        variances = tf.math.reduce_variance(act_normalized, axis=[0, 1])
        top_indices = tf.argsort(variances, direction='DESCENDING')[:max_channels]
        
        scores = np.zeros(max_channels)
        
        # 4. Batched Scoring
        for i in tqdm(range(0, len(top_indices), batch_size), desc="Batch ScoreCAM", leave=False):
            end_idx = min(i + batch_size, len(top_indices))
            current_indices = top_indices[i:end_idx]
            
            # Gather only the top channels and prepare for broadcasting
            # maps shape: (num_in_batch, H, W, 1)
            maps = tf.gather(act_normalized, current_indices, axis=-1)
            maps = tf.transpose(maps, [2, 0, 1])
            maps = tf.expand_dims(maps, axis=-1)
            
            # Create masked batch: (batch_size, 260, 260, 3)
            # x is (1, 260, 260, 3), broadcasting handles the expansion
            masked_batch = x * maps
            
            # Single forward pass for the whole batch
            preds = self.model(masked_batch, training=False)
            
            # The score_fn must handle batch logic
            # Modified score_fn logic:
            batch_scores = score_fn(preds)
            scores[i:end_idx] = batch_scores.numpy()

        # 5. Weighted Sum using only the top channels
        cam = np.zeros(input_shape.numpy(), dtype=np.float32)
        # Normalize scores
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        
        for idx, weight in enumerate(scores):
            channel_idx = top_indices[idx]
            cam += weight * act_normalized[..., channel_idx].numpy()
        
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-10)
        return cam

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    
    if not PREDICTIONS_CSV.exists():
        print(f"Error: Could not find predictions CSV at {PREDICTIONS_CSV}")
        return

    df = pd.read_csv(PREDICTIONS_CSV)
    
    if TEST_ONLY_TB_POSITIVE:
        print("Filtering dataset to ONLY include TB Positive cases...")
        df = df[df['true_label'] == 1]
    
    if TEST_MODE_LIMIT is not None:
        print(f"--- TEST MODE ACTIVE: Only running on {TEST_MODE_LIMIT} images ---")
        df = df.head(TEST_MODE_LIMIT)
    else:
        print(f"--- FULL RUN ACTIVE: Processing all {len(df)} images ---")

    grouped = df.groupby('fold')

    for fold, fold_df in grouped:
        print(f"\nProcessing Fold {fold} ({len(fold_df)} images)...")
        
        fold_out_dir = VISUALIZATION_DIR / f"fold_{fold}"
        fold_out_dir.mkdir(exist_ok=True)

        model = build_efficientnet_model()
        weight_path = WEIGHTS_DIR / f"fold_{fold}.weights.h5"
        
        if not weight_path.exists():
            print(f"Weights missing for Fold {fold}. Skipping...")
            continue
            
        model.load_weights(weight_path)

        # Disable augmentation by setting the RandomFlip layer to non‑trainable
        # and we will call everything with training=False.
        model.layers[0].trainable = False

        scorecam = ScoreCAM(model, target_layer_name='top_conv')
        for _, row in fold_df.iterrows():
            unet_path_str = row['path']
            true_label = row['true_label']
            pred_prob = row['predicted_probability']
            pred_label = row['predicted_label']
            
            # 1. Load and Preprocess Images
            seg_bgr = cv2.imread(unet_path_str)
            if seg_bgr is None: continue
            seg_rgb = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2RGB)
            seg_resized = cv2.resize(seg_rgb, (IMG_SIZE, IMG_SIZE))
            
            # Load original X-ray for overlay
            unet_path = Path(unet_path_str)
            source_dir = unet_path.parent.name
            orig_stem = unet_path.name.replace("_unetseg.png", ".png")
            orig_img_path = (SHENZHEN_CXR_DIR if source_dir == "shenzhen" else MONTGOMERY_CXR_DIR) / orig_stem
            
            orig_bgr = cv2.imread(str(orig_img_path))
            orig_resized = cv2.resize(cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE)) if orig_bgr is not None else seg_resized

            # Preprocess for model
            x_img = np.expand_dims(seg_resized, axis=0).astype(np.float32)
            x_img = tf.keras.applications.efficientnet.preprocess_input(x_img)

            # 2. Define Batch-Aware Score Function
            # We use a default argument (p_lab=pred_label) to "freeze" the label for this specific loop iteration
            def score_fn(output, p_lab=pred_label):
                return output[:, 0] if p_lab == 1 else -output[:, 0]

            # 3. Generate Heatmap (THE FAST WAY)
            heatmap = scorecam(score_fn, x_img, batch_size=16, max_channels=256)
            
            # 4. Apply colormap and blend
            heat_rgb = (cm.jet(heatmap)[..., :3] * 255).astype(np.float32)
            lung_mask = (np.max(seg_resized, axis=-1, keepdims=True) > 10).astype(np.float32)
            heatmap_expanded = np.expand_dims(heatmap, axis=-1)
            
            # Only show high-confidence areas (thresholding)
            alpha_map = np.clip((heatmap_expanded - 0.25) / 0.75, 0.0, 1.0) * 0.65
            final_alpha = alpha_map * lung_mask
            
            orig_float = orig_resized.astype(np.float32)
            blended_img = (heat_rgb * final_alpha + orig_float * (1.0 - final_alpha)).astype(np.uint8)
            
            # 5. Save figure
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(orig_resized)
            ax[0].set_title(f"Original\nTrue: {true_label}")
            ax[0].axis('off')
            
            ax[1].imshow(blended_img)
            misclass = " [MISCLASSIFIED]" if true_label != pred_label else ""
            ax[1].set_title(f"ScoreCAM\nPred: {pred_label} ({pred_prob:.3f}){misclass}")
            ax[1].axis('off')
            
            plt.tight_layout()
            save_path = fold_out_dir / f"{Path(orig_img_path).stem}_scorecam.png"
            plt.savefig(save_path, dpi=150)
            plt.close(fig)
            print(f"  Saved: {save_path}")

    print("\nScoreCAM Generation Complete!")

if __name__ == "__main__":
    main()