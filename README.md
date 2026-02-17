# Pneumonia Detection App

This project is a Flask-based web application that uses deep learning models to detect pneumonia from X-ray images. It includes three different models for validation, binary classification, and multiclass classification.

## Models

The application utilizes the following Keras models:
- **X-Ray Detector (Validation):** `xray_detector.keras` - Ensures the uploaded image is an X-ray.
- **Binary Classification:** `efficientnet_binary_robust.keras` - Classifies X-rays as Normal or Pneumonia.
- **Multi-Class Classification:** `efficientnet_b0_optimized_v7.keras` - Further classifies pneumonia types (Optimized V7).

## Features
- Web-based interface for image upload and prediction.
- Robust detection using EfficientNet architectures.
- Multi-stage validation to ensure input quality.

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ghulammustafatariq/Pneumonia-Detection.git
   cd Pneumonia-Detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Ensure you have TensorFlow, Flask, Pillow, and OpenCV installed.)*

3. **Run the application:**
   ```bash
   python app.py
   ```

## Repository Structure
- `app.py`: Main Flask application.
- `templates/`: HTML templates for the web interface.
- `Finalized model/`: Jupyter notebooks containing training logic and experiments.
- `*.keras`: Pre-trained model files.
- `Slides/`: Project presentation materials.
