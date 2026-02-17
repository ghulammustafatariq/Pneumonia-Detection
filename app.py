import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras import Sequential
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
import cv2
import base64
from tensorflow.keras.models import Model
from tensorflow.keras import Input
import io
import os

# --- Flask App Initialize ---
app = Flask(__name__)

# --- 1. Model Paths ---
PATH_MODEL_VAL = 'xray_detector.keras'
PATH_MODEL_BIN = 'efficientnet_binary_robust.keras' 
PATH_MODEL_MULTI = "efficientnet_b0_optimized_v7.keras"

# --- 2. Load Models ---
print("--- Loading Models... ---")

def build_val_model():
    base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights=None)
    base_model.trainable = False
    model = Sequential([
        base_model, GlobalAveragePooling2D(), Dropout(0.2), Dense(1, activation='sigmoid')
    ])
    return model

try:
    model_val = build_val_model()
    model_val.load_weights(PATH_MODEL_VAL)
    print("✅ Model 1 (Validation) Loaded.")
    
    model_bin = load_model(PATH_MODEL_BIN, compile=False)
    print("✅ Model 2 (Binary EfficientNet) Loaded.")
    
    model_multi = load_model(PATH_MODEL_MULTI, compile=False)
    print("✅ Model 3 (Multi-Class EffNet V7) Loaded.")

except Exception as e:
    print(f"❌ Critical Error loading models: {e}")
    
CLASS_NAMES_MULTI = ['COVID-19', 'Lung Opacity', 'Normal', 'Pneumonia (Bacterial)', 'Pneumonia (Viral)', 'Tuberculosis']

# --- 3. Preprocessing Logic ---
def preprocess_image(img_bytes, target_size, model_type):
    try:
        img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Resize
        img_pil = img_pil.resize(target_size, Image.LANCZOS)
        img_array_original = np.array(img_pil)
        
        # Create batch axis
        img_array_for_model = np.expand_dims(img_array_original.copy(), axis=0)
        
        preprocessed_array = None
        
        if model_type == 'mobilenet':
            preprocessed_array = mobilenet_preprocess(img_array_for_model.astype(np.float32))
            
        elif model_type == 'efficientnet':
            preprocessed_array = efficientnet_preprocess(img_array_for_model)
            
        elif model_type == 'efficientnet_v7':
            preprocessed_array = efficientnet_preprocess(img_array_for_model)
            
        return img_array_original, preprocessed_array
        
    except Exception as e:
        print(f"Preprocessing Error: {e}")
        return None, None

# --- 4. Uncertainty Estimation (New Feature) ---
def predict_with_uncertainty(model, img_tensor, runs=20):
    """
    Aggressive TTA to force variance if model is truly uncertain.
    """
    # 1. Batch Create
    batch = tf.repeat(img_tensor, repeats=runs, axis=0)
    
    # 2. Aggressive Augmentations (Changed from 320 to 340 padding for bigger crops)
    # Ziyaada zoom/shift taake model ko challenge ho
    batch = tf.image.resize_with_crop_or_pad(batch, 340, 340) 
    batch = tf.image.random_crop(batch, size=(runs, 300, 300, 3))
    
    # Brightness/Contrast range barha diya (0.05 -> 0.15)
    batch = tf.image.random_brightness(batch, 0.15) 
    batch = tf.image.random_contrast(batch, 0.8, 1.2)
    
    # 3. Predict
    preds = model.predict(batch, verbose=0)
    
    # 4. Stats
    mean_preds = np.mean(preds, axis=0)
    variance_preds = np.var(preds, axis=0)
    
    return mean_preds, variance_preds

# --- 5. Validation Helper ---
def validate_image(img_bytes):
    _, img_val = preprocess_image(img_bytes, (128, 128), 'mobilenet')
    if img_val is None: return False, "Preprocessing failed"
    pred = model_val.predict(img_val)[0][0]
    if pred > 0.55: return False, f'Not an X-Ray (Score: {pred:.4f})'
    return True, "Valid X-Ray"

# --- 6. Grad-CAM Function ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except:
        # Fallback
        last_conv_layer = model.get_layer("top_activation") 

    last_conv_out = last_conv_layer.output
    if isinstance(last_conv_out, list): last_conv_out = last_conv_out[0]

    grad_model = Model(inputs=model.inputs, outputs=[last_conv_out, model.output])

    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        (conv_outputs, predictions) = grad_model(inputs)
        
        if isinstance(conv_outputs, list): conv_outputs = conv_outputs[0]
        if isinstance(predictions, list): predictions = predictions[0]
        
        top_pred_index = tf.argmax(predictions[0])
        top_class_channel = predictions[:, top_pred_index]

    grads = tape.gradient(top_class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    
    max_val = np.max(heatmap)
    if max_val == 0: max_val = 1e-10
    heatmap /= max_val
    
    return heatmap.astype(np.float32)

def create_gradcam_overlay(img_original_rgb, heatmap, alpha=0.4):
    img_bgr = cv2.cvtColor(img_original_rgb, cv2.COLOR_RGB2BGR)
    heatmap = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap_color * alpha + img_bgr * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    _, buffer = cv2.imencode('.jpg', superimposed_img)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

# --- 7. Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_binary', methods=['POST'])
def predict_binary():
    if 'image' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['image']
    img_bytes = file.read()

    try:
        is_valid, msg = validate_image(img_bytes)
        if not is_valid: return jsonify({'error': msg}), 400

        img_original, img_bin = preprocess_image(img_bytes, (224, 224), 'efficientnet')
        
        pred_prob = model_bin.predict(img_bin)[0][0]
        prob_pneumonia = float(pred_prob)
        prob_normal = 1.0 - prob_pneumonia
        
        results = {'Normal': round(prob_normal*100, 2), 'Pneumonia': round(prob_pneumonia*100, 2)}

        gradcam_base64 = None
        try:
            heatmap = make_gradcam_heatmap(img_bin, model_bin, "top_activation")
            gradcam_base64 = create_gradcam_overlay(img_original, heatmap)
        except Exception as e:
            print(f"Grad-CAM Error: {e}")

        return jsonify({'status': 'success', 'validation': msg, 'binary_prediction': results, 'gradcam_image': gradcam_base64})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_multiclass', methods=['POST'])
def predict_multiclass():
    if 'image' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['image']
    img_bytes = file.read()

    try:
        # 1. Validate
        is_valid, msg = validate_image(img_bytes)
        if not is_valid: return jsonify({'error': msg}), 400

        # 2. Preprocess
        img_original, img_multi = preprocess_image(img_bytes, (300, 300), 'efficientnet_v7')
        
        # 3. PREDICT WITH UNCERTAINTY (20 Runs)
        # Instead of model.predict(), we call our custom function
        mean_preds, var_preds = predict_with_uncertainty(model_multi, img_multi, runs=20)
        
        # 4. Format Results
        # We pass both Mean Confidence and Variance
        results = {}
        uncertainty_info = {}
        
        for i, p in enumerate(mean_preds):
            class_name = CLASS_NAMES_MULTI[i]
            results[class_name] = round(float(p)*100, 2)
            # Scaling variance for display (x1000 makes it readable)
            uncertainty_info[class_name] = round(float(var_preds[i])*1000, 4)

        # 5. Grad-CAM
        gradcam_base64 = None
        try:
            heatmap = make_gradcam_heatmap(img_multi, model_multi, "top_activation")
            gradcam_base64 = create_gradcam_overlay(img_original, heatmap)
        except Exception as e:
            print(f"Grad-CAM Error: {e}")

        return jsonify({
            'status': 'success', 
            'validation': msg, 
            'multiclass_prediction': results, 
            'uncertainty': uncertainty_info, # Sending Variance Data
            'gradcam_image': gradcam_base64
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)