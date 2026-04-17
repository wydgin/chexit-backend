import os
# Important: Your scorecam script needs this set before TF loads
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import numpy as np
import cv2
import io
from PIL import Image

# Absolute imports from your app package
from app.unet_segmentation import dice_coef, dice_coef_loss, iou_coef
from app.generate_scorecam import build_efficientnet_model # Use your existing builder

# Constants
IMG_SIZE_UNET = 256  
IMG_SIZE_EFFNET = 260 

def load_unet():
    custom_objects = {
        "dice_coef": dice_coef,
        "dice_coef_loss": dice_coef_loss,
        "iou_coef": iou_coef,
    }
    # Using the 'best' file we identified
    return tf.keras.models.load_model('app/models/unet_lung_seg_best.h5', custom_objects=custom_objects)

# Load models once when the server starts to save time
unet_model = load_unet()

effnet_folds = []
for i in range(1, 6):
    # Use the model builder from your actual script to ensure layers match perfectly
    m = build_efficientnet_model() 
    m.load_weights(f'app/models/fold_{i}.weights.h5')
    effnet_folds.append(m)

def run_tb_ensemble(image_bytes):
    # --- STEP 1: SEGMENTATION ---
    nparr = np.frombuffer(image_bytes, np.uint8)
    orig_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Preprocess for U-Net
    img_gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    resized_gray = cv2.resize(img_gray, (IMG_SIZE_UNET, IMG_SIZE_UNET)).astype("float32") / 255.0
    
    # Predict Mask
    # Expand dims to (1, 256, 256, 1)
    pred_mask = unet_model.predict(np.expand_dims(resized_gray, [0, -1]), verbose=0)[0]
    mask_binary = (pred_mask > 0.5).astype(np.uint8)
    
    # Apply mask to original image
    mask_resized = cv2.resize(mask_binary, (orig_img.shape[1], orig_img.shape[0]))
    segmented_img = cv2.bitwise_and(orig_img, orig_img, mask=mask_resized)

    # --- STEP 2: PREDICTION ---
    # Prepare for EfficientNet
    eff_input = cv2.resize(segmented_img, (IMG_SIZE_EFFNET, IMG_SIZE_EFFNET))
    # EfficientNet usually expects RGB
    eff_input_rgb = cv2.cvtColor(eff_input, cv2.COLOR_BGR2RGB)
    eff_input_final = np.expand_dims(eff_input_rgb, axis=0)
    # Note: EfficientNetV2/B2 preprocess is usually just scaling or built-in
    
    # Ensemble Average
    probs = [float(m.predict(eff_input_final, verbose=0)[0][0]) for m in effnet_folds]
    avg_prob = sum(probs) / len(probs)

    # --- STEP 3: SCORE-CAM ---
    # Placeholder: You can now call your actual Score-CAM function here
    # from app.generate_scorecam import your_function_name
    # heatmap = your_function_name(effnet_folds[3], eff_input_final)
    heatmap_dummy = np.zeros((IMG_SIZE_EFFNET, IMG_SIZE_EFFNET, 3)) 

    return avg_prob, heatmap_dummy