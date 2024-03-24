from flask import Blueprint, render_template, request, send_from_directory
from .app_functions import ValuePredictor, pred
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np

prediction = Blueprint('prediction', __name__)

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

dir_path = os.path.dirname(os.path.realpath(__file__))


def preprocess_image(image_path, target_size):
    import cv2
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = np.array(image) / 255.0  # Normalize pixel values
    return image

from tensorflow.keras.models import load_model

# Load the model
model = load_model('/Users/rakeshcavala/Desktop/hc/website/app_models/Lung_Model.h5')


@prediction.route('/lung_cancer', methods=['POST'])
def lung_cancer():
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

        predictions = model.predict(processed_image)
        # Get the predicted class label
        predicted_class = np.argmax(predictions, axis=1)[0]  # Selecting the first (and only) element

# Define class labels
        class_labels = ['Normal', 'Malignant', 'Benign']

# Get the predicted class name
        prediction = class_labels[predicted_class]

        
        # Remove the temporary image file
        
        
        # Return the prediction result to the user
        return render_template('lr_result.html', prediction=prediction, image = image_path)

@prediction.route('/predict', methods=["POST", 'GET'])
def predict():

    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        result, page = ValuePredictor(to_predict_list) 
        return render_template("result.html", prediction=result, page=page)
    else:
        return render_template( 'base.html')

@prediction.route('/upload', methods=['POST','GET'])
def upload_file():
    if request.method=="GET":
        return render_template('pneumonia.html', title='Pneumonia Disease')
    else:
        file = request.files["file"]
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath,'uploads',  secure_filename(file.filename))
        file.save(file_path)
        indices = {0: 'Normal', 1: 'Pneumonia'}
        result = pred(file_path)

        if result>0.5:
            label = indices[1]
            accuracy = result * 100
        else:
            label = indices[0]
            accuracy = 100 - result
        return render_template('deep_pred.html', image_file_name=file.filename, label = label, accuracy = accuracy)

@prediction.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)