import numpy as np

def run_tb_ensemble(image_bytes):
    """
    Placeholder for your ensemble. 
    Input: Raw image bytes
    Output: Probability, Heatmap Image (numpy array)
    """
    # TODO: Load your ensemble models here
    # 1. Preprocess image_bytes
    # 2. Run MobileNetV4, EfficientNetB2, DenseNet-121
    # 3. Apply Score-CAM
    
    # Mock data for now so you can test the API
    mock_probability = 0.85
    mock_heatmap = np.zeros((224, 224, 3)) # Black image placeholder
    
    return mock_probability, mock_heatmap