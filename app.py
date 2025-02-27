import os
import numpy as np
import cv2
import sqlite3
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 
import base64
from io import BytesIO
import json
import requests
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from werkzeug.utils import secure_filename
from datetime import datetime
import google.generativeai as genai



app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)


# Class names for the classification model
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Database initialization
def init_db():
    conn = sqlite3.connect('brain_mri.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        original_path TEXT NOT NULL,
        result_path TEXT NOT NULL,
        classification TEXT NOT NULL,
        confidence REAL NOT NULL,
        summary TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()
    conn.close()

# Helper functions for model inference
def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def iou(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    total = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f)
    union = total - intersection
    return (intersection + smooth) / (union + smooth)

# Load the models
classification_model = load_model('brain_mri.h5', compile=False)
segmentation_model = load_model('Unet_model.h5', custom_objects={
    'dice_coefficient': dice_coefficient,
    'dice_loss': dice_loss,
    'iou': iou
})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image_for_classification(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def preprocess_image_for_segmentation(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    return np.expand_dims(img, axis=(0, -1))



# configure genai api
genai.configure(api_key='AIzaSyDsk7uew0pRUr-jaYABqyUxdqpY8sLGi18')
model = genai.GenerativeModel("gemini-1.5-flash")
def get_bioGPT_summary(classification, confidence):
    # This is a placeholder. In a real application, you would use BioGPT's API
    # For now, we'll return a simple template-based summary
    summaries = {
        'glioma': "Glioma is a type of tumor that occurs in the brain and spinal cord. Gliomas begin in the glial cells that surround and support nerve cells. The treatment and prognosis depend on the grade and location of the tumor.",
        'meningioma': "Meningioma is a tumor that forms on membranes that cover the brain and spinal cord just inside the skull. Most meningiomas are noncancerous, though rarely some can be cancerous. Treatment options include surgery, radiation therapy, and regular monitoring.",
        'no_tumor': "No evidence of tumor detected in the MRI scan. Regular follow-up may still be recommended as per standard medical protocols.",
        'pituitary': "Pituitary tumors are abnormal growths that develop in the pituitary gland at the base of the brain. Most pituitary tumors are noncancerous and don't spread to other parts of the body. Treatment options include surgery, radiation therapy, and medication."
    }
    
    confidence_statement = ""
    if confidence > 0.9:
        confidence_statement = "The model has high confidence in this classification."
    elif confidence > 0.7:
        confidence_statement = "The model has moderate confidence in this classification."
    else:
        confidence_statement = "The model has low confidence in this classification. Consider seeking a second opinion."
    
    return f"{summaries.get(classification, 'Unknown tumor type')} {confidence_statement}"  
    
    
    return response.text if response else "Error generating summary."

def save_to_database(filename, original_path, result_path, classification, confidence, summary):
    conn = sqlite3.connect('brain_mri.db')
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO analyses (filename, original_path, result_path, classification, confidence, summary)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (filename, original_path, result_path, classification, confidence, summary))
    analysis_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return analysis_id

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'mri_image' not in request.files:
        flash('No file part', 'error')
        return redirect(request.url)
    
    file = request.files['mri_image']
    
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Save original image
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_filename = f"{timestamp}_{filename}"
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        file.save(original_path)
        
        # Classify the image
        classification_input = preprocess_image_for_classification(original_path)
        predictions = classification_model.predict(classification_input)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])
        
        # Segment the image
        segmentation_input = preprocess_image_for_segmentation(original_path)
        segmentation_mask = segmentation_model.predict(segmentation_input)
        segmentation_mask = (segmentation_mask > 0.5).astype(np.uint8)
        
        # Create overlay image
        plt.figure(figsize=(10, 8))
        
        # Original image
        img = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (128, 128))
        
        # Display original and mask overlay
        plt.subplot(1, 2, 1)
        plt.imshow(img_resized, cmap='gray')
        plt.title('Original MRI')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(img_resized, cmap='gray')
        plt.imshow(segmentation_mask[0, :, :, 0], alpha=0.5, cmap='jet')
        plt.title('Tumor Segmentation')
        plt.axis('off')
        
        # Save the result
        result_filename = f"result_{saved_filename.split('.')[0]}.png"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        plt.savefig(result_path, bbox_inches='tight')
        plt.close()
        
        # Get a summary from Gemini (simulated)
        summary = get_bioGPT_summary(predicted_class, confidence)
        
        # Save to database
        analysis_id = save_to_database(
            saved_filename, 
            original_path, 
            result_path, 
            predicted_class,
            confidence,
            summary
        )
        
        # Redirect to results page
        return redirect(url_for('result', analysis_id=analysis_id))
    
    flash('Invalid file type. Please upload JPG, JPEG, or PNG files.', 'error')
    return redirect(request.url)

@app.route('/result/<int:analysis_id>')
def result(analysis_id):
    conn = sqlite3.connect('brain_mri.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM analyses WHERE id = ?', (analysis_id,))
    analysis = cursor.fetchone()
    conn.close()
    
    if analysis:
        return render_template('result.html', analysis=analysis)
    else:
        flash('Analysis not found', 'error')
        return redirect(url_for('index'))

@app.route('/history')
def history():
    conn = sqlite3.connect('brain_mri.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM analyses ORDER BY created_at DESC LIMIT 20')
    analyses = cursor.fetchall()
    conn.close()
    
    return render_template('history.html', analyses=analyses)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)