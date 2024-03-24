# Import necessary modules and packages
from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from PIL import Image
import tensorflow as tf
import pandas as pd
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import spacy
from spacy import displacy
from flaskext.markdown import Markdown
from dotenv.main import load_dotenv
from flask_bcrypt import generate_password_hash, check_password_hash

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Enable CORS
CORS(app)

# Load spaCy models for medical entity recognition
nlp = spacy.load("en_core_web_sm")
nlq = spacy.load("en_ner_bionlp13cg_md")
nle = spacy.load("en_ner_bc5cdr_md")

# Load TensorFlow models for cancer detection
VGG16_model = tf.keras.models.load_model("/path/to/VGG16_model.h5")
tumor_model = tf.keras.models.load_model("/path/to/brain_tumor_detector.h5")
lung_cancer_model = load_model('/path/to/Lung_Model.h5')

# Load clinical trial data
df = pd.read_csv('/path/to/ctg-studies.csv')

# Define HTML wrapper for rendering spaCy visualizations
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

# Define routes for various functionalities

# Clinical trial search
@app.route("/access-application")
def access_application():
    """
    Route for accessing the clinical trial search application.
    """
    return render_template('search.html')

@app.route("/cltm")
def cltm():
    """
    Route for accessing the clinical trial management page.
    """
    return render_template('clinical.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    """
    Route for handling the clinical trial search functionality.
    """
    # Handle search functionality
    # ...
    return render_template('r.html', trials=trials)

# Medical entity recognition
@app.route("/entities")
def ent():
    """
    Route for accessing the medical entity recognition page.
    """
    return render_template('ner.html')

@app.route("/biomedical_info")
def ents():
    """
    Route for accessing the biomedical information page.
    """
    return render_template('medical copy 2.html')

@app.route('/extractss',methods=["GET","POST"])
def extractss():
    """
    Route for extracting medical entities from text.
    """
    if request.method == 'POST':
        raw_text = request.form['rawtext']
        docx = nlq(raw_text)
        html = displacy.render(docx,style="ent")
        html = html.replace("\n\n","\n")
        result = HTML_WRAPPER.format(html)
    return render_template('rest.html',rawtext=raw_text,result=result)

# Skin cancer detection
@app.route('/skin_cancer_detection', methods=['POST'])
def skin_cancer_detection():
    """
    Route for detecting skin cancer from uploaded images.
    """
    if 'imageFile' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['imageFile']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Save the uploaded image
    image_path = "uploaded_image.jpg"
    file.save(image_path)
    
    # Make prediction using VGG16 model
    vgg16_prediction = predict_skin_cancer(image_path, VGG16_model)
    
    # You can choose which model's prediction to return based on your requirements
    result = vgg16_prediction  # Or you can choose vgg16_prediction
    
    return jsonify({'result': result, 'image': image_path})

# Brain tumor detection
@app.route('/brain_tumor_detection', methods=['POST'])
def tumor():
    """
    Route for detecting brain tumors from uploaded images.
    """
    import cv2
    if request.method == 'POST':
        # Get the uploaded image file
        file = request.files['file']
        
        # Save the uploaded image temporarily
        image_path = 'temp_image.jpg'
        file.save(image_path)
        
        # Preprocess the uploaded image
        target_size = (224, 224)  # Update with the input dimensions of your model
        processed_image = preprocess_image(image_path, target_size)
        
        # Reshape the image to match the input shape expected by the model
        processed_image = np.expand_dims(processed_image, axis=0)
        
        # Make predictions
        predictions = tumor_model.predict(processed_image)
        
        # Get the predicted class label
        predicted_class = np.round(predictions)
        
        # Define class labels
        if predicted_class==0:
            prediction = "healthy"
        if predicted_class==1:
            prediciton = "tumor"
        
        # Return the prediction result to the user
        return render_template('rt.html', prediction=prediction)

# Lung cancer detection
@app.route('/lung_cancer', methods=['POST'])
def lung_cancer():
    """
    Route for detecting lung cancer from uploaded images.
    """
    import cv2
    if request.method == 'POST':
        # Get the uploaded image file
        file = request.files['file']
        
        # Save the uploaded image temporarily
        image_path = 'temp_image.jpg'
        file.save(image_path)
        
        # Preprocess the uploaded image
        target_size = (244, 244)  # Update with the input dimensions of your model
        processed_image = preprocess_image(image_path, target_size)
        
        # Reshape the image to match the input shape expected by the model
        processed_image = np.expand_dims(processed_image, axis=0)
        
        # Make predictions
        predictions = lung_cancer_model.predict(processed_image)
        
        # Get the predicted class label
        predicted_class = np.argmax(predictions)
        
        # Define class labels
        class_labels = ['Malignant', 'Normal', 'Benign']
        
        # Get the predicted class name
        prediction = class_labels[predicted_class]
        
        # Remove the temporary image file
        os.remove(image_path)
        
        # Return the prediction result to the user
        return render_template('lr_result.html', prediction=prediction, image=image_path)

# Chatbot functionality
@app.route('/chatbot', methods=['POST'])
def get_data():
    """
    Route for interacting with the chatbot.
    """
    data = request.get_json()
    text = data.get('data')
    user_input = text
    try:
        conversation = ConversationChain(llm=llm, memory=memory)
        output = conversation.predict(input=user_input)
        memory.save_context({"input": user_input}, {"output": output})
        return jsonify({"response": True, "message": output})
    except Exception as e:
        print(e)
        error_message = f'Error: {str(e)}'
        return jsonify({"message": error_message, "response": False})

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
