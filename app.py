
from website import create_app
from website.views import views  # Assuming your blueprint is named 'views'
import cv2
from flask import Blueprint, render_template
from flask import Flask, render_template,jsonify,request
from flask_cors import CORS
import requests,openai,os
from dotenv.main import load_dotenv
from langchain_community.chat_models import ChatOpenAI  # Import ChatOpenAI class
#from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from flask_bcrypt import generate_password_hash, check_password_hash
from flask import Flask, render_template, jsonify, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
import random
from werkzeug.utils import secure_filename
from transcription_service import TranscriptionService
from summarization_service import SummarizationService
import os
import spacy
from spacy import displacy
nlp = spacy.load("en_core_web_sm")
import json
from spacy import displacy
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd




HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

from flaskext.markdown import Markdown

from markupsafe import Markup

app = create_app()
Markdown(app)

df = pd.read_csv('/Users/rakeshcavala/Downloads/ctg-studies (1).csv')


@app.route("/access-application")
def access_application():
    return render_template('search.html')


@app.route("/cltm")
def cltm():
    return render_template('clinical.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        age_category = request.form['age_category']
        sex = request.form['sex']
        conditions = request.form['conditions'].split(',')
        study_type = request.form['study_type']
        

        # Assuming your DataFrame has a column 'Age' that contains values like 'CHILD', 'ADULT', 'OLDER_ADULT'
        # Filter DataFrame based on user inputs
        df['Sex'] = df['Sex'].fillna('').str.lower()
        df['Conditions'] = df['Conditions'].fillna('').str.lower()
        
        df['Age'] = df['Age'].fillna('').str.lower()
        filtered_df = df[df['Age'].str.contains(age_category, case=False)]
        filtered_df = filtered_df[filtered_df['Sex'].str.contains(sex, case=False)]

        for x in filtered_df['Conditions']:
             for condition in conditions:
                  if condition.lower() in x.lower():
                       filtered_df
             
        filtered_df = filtered_df[filtered_df['Conditions'].apply(lambda x: all(condition.lower() in x.lower() for condition in conditions))]
        filtered_df = filtered_df[filtered_df['Study Type'].apply(lambda x: any(study_types in x for study_types in study_type))]
        
             

        # Convert filtered DataFrame to a list of dicts to pass to the frontend
        trials = filtered_df.to_dict('records')
        return render_template('r.html', trials=trials)

    return render_template('search.html')



def visualize_entities(doc):
    html = displacy.render(doc, style="ent", page=True)

    return html



@app.route('/trans')
def trans():
    return render_template('tren.html')



@app.route('/extract',methods=["GET","POST"])
def extract():
    
	if request.method == 'POST':
        
		raw_text = request.form['rawtext']
		docx = nlp(raw_text)
		html = displacy.render(docx,style="ent")
		html = html.replace("\n\n","\n")
		result = HTML_WRAPPER.format(html)

	return render_template('rest.html',rawtext=raw_text,result=result)




app.config['UPLOAD_FOLDER'] = 'uploads'  # Make sure this folder exists
app.config['UPLOAD_EXTENSIONS'] = ['.m4a','.mp3','.webm','.mp4','.mpga','.wav','.mpeg']


if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

transcription_service = TranscriptionService()
summarization_service = SummarizationService()


# Prompts for different video/audio types
summarization_prompt = "You will analyze a huge transcript from a clinical video or clinical audio create notes for doctor in a form of list. Translate in English if needed"
important_dates_prompt = "You will analyze a huge data create a summary of all mentioned important dates in the form of a list and also extraxt all the medical entinties and give it in a list format"
meeting_prompt = "You will analyze a huge data create a summary of all mentioned important dates in the form of a list and also extraxt all the medical entinties and give it in a list format"




# Register the 'views' blueprint with a unique name


app.register_blueprint(views, name='main_views')
llm = ChatOpenAI(openai_api_key='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', model_name="gpt-3.5-turbo")
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)



VGG16_model = tf.keras.models.load_model("/Users/rakeshcavala/Desktop/hc/website/app_models/VGG16_model.h5")


def preproces_image(image_path, target_size):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0


def predict_skin_cancer(image_path, model):
    img_array = preproces_image(image_path, target_size=(224, 224))  
    prediction = model.predict(img_array)
    
    return "Benign" if prediction[0][0] > 0.5 else "Malignant"


@app.route('/skin_cancer_detection', methods=['POST'])
def skin_cancer_detection():
    if 'imageFile' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['imageFile']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Save the uploaded image
    image_path = "uploaded_image.jpg"
    file.save(image_path)
    
    # Make prediction using ResNet50 model
    
    
    # Make prediction using VGG16 model
    vgg16_prediction = predict_skin_cancer(image_path, VGG16_model)
    
    # You can choose which model's prediction to return based on your requirements
    result = vgg16_prediction  # Or you can choose vgg16_prediction
    
    return jsonify({'result': result, 'image': image_path})



@app.route("/entities")
def ent():
    return render_template('ner.html')

@app.route("/biomedical_info")
def ents():
    return render_template('medical copy 2.html')

nlq = spacy.load("en_ner_bionlp13cg_md")

@app.route('/extractss',methods=["GET","POST"])
def extractss():
    
	if request.method == 'POST':
        
		raw_text = request.form['rawtext']
		docx = nlq(raw_text)
		html = displacy.render(docx,style="ent")
		html = html.replace("\n\n","\n")
		result = HTML_WRAPPER.format(html)

	return render_template('rest.html',rawtext=raw_text,result=result)

@app.route("/chemical_disease_info")
def entu():
    return render_template('medical copy 3.html')

nle = spacy.load("en_ner_bc5cdr_md")

@app.route('/extractsss',methods=["GET","POST"])
def extractsss():
    
	if request.method == 'POST':
        
		raw_text = request.form['rawtext']
		docx = nle(raw_text)
		html = displacy.render(docx,style="ent")
		html = html.replace("\n\n","\n")
		result = HTML_WRAPPER.format(html)

	return render_template('rest.html',rawtext=raw_text,result=result)







@app.route("/general_info")
def gen():
    return render_template("medical.html")

@app.route("/drug_info")
def drug():
    return render_template("medical copy.html")


@app.route('/extracts',methods=["GET","POST"])
def extracts():
    
	if request.method == 'POST':
        
		raw_text = request.form['rawtext']
		docx = nlp(raw_text)
		html = displacy.render(docx,style="ent")
		html = html.replace("\n\n","\n")
		result = HTML_WRAPPER.format(html)

	return render_template('rest.html',rawtext=raw_text,result=result)


@app.route('/op')
def op():
    return render_template('index.html')


tumor_model = tf.keras.models.load_model("/Users/rakeshcavala/Desktop/hc/website/app_models/brain_tumor_detector.h5")

@app.route('/brain_tumor_detection', methods=['POST'])
def tumor():
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


    


@app.route('/run', methods=['POST'])
def run_the_thing():

    # Collecting input from flask html
    # Conditions for determining language
    if request.form['whisper_model'] == "large":
        whisper_model = "large"
    elif request.form['language'] == "English":
        whisper_model = request.form['whisper_model']+".en"
    else:
        whisper_model = request.form['whisper_model']

    video_type = request.form['video_type']
    gpt_model = request.form['gpt_model']
    api_key = "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"

    # For terminal view
    print("Whisper Model: " + whisper_model)
    print("GPT Model: " + gpt_model)

    # Handle file upload
    file = request.files.get('file')
    s_filename = secure_filename(file.filename)
    if s_filename != '':
        file_ext = os.path.splitext(s_filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            return render_template('index.html', summary="File type not supported. Please select one of: .m4a .mp3 .webm .mp4 .mpga .wav .mpeg")
    
    
    if file and file.filename != '':
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        link = file_path

    transcript = transcription_service.transcribe(link, whisper_model, api_key)
    summary = summarization_service.summarize(
        transcript, gpt_model, api_key, summarization_prompt)

    os.remove(file_path)

    # Conditions for selecting video type
    match video_type:
        case "lecture":
            important_dates = summarization_service.summarize(
                transcript, gpt_model, api_key, important_dates_prompt)
            return render_template('index.html', summary=summary, important_dates=important_dates, transcript=transcript)
        case "meeting":
            meeting_points = summarization_service.summarize(
                transcript, gpt_model, api_key, meeting_prompt)
            return render_template('index.html', summary=summary, important_dates=meeting_points, transcript=transcript)
        case _:
            return render_template('index.html', summary=summary, transcript=transcript)




    
@app.route('/chatbot', methods=['POST'])
def get_data():
    data = request.get_json()
    text=data.get('data')
    user_input = text
    try:
        conversation = ConversationChain(llm=llm,memory=memory)
        output = conversation.predict(input=user_input)
        memory.save_context({"input": user_input}, {"output": output})
        return jsonify({"response":True,"message":output})
    except Exception as e:
        print(e)
        error_message = f'Error: {str(e)}'
        return jsonify({"message":error_message,"response":False})



def preprocess_image(image_path, target_size):
    import cv2
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = np.array(image) / 255.0  # Normalize pixel values
    return image

from tensorflow.keras.models import load_model

# Load the model
model = load_model('/Users/rakeshcavala/Desktop/hc/website/app_models/Lung_Model.h5')


@app.route('/lung_cancer', methods=['POST'])
def lung_cancer():
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
        predictions = model.predict(processed_image)
        
        # Get the predicted class label
        predicted_class = np.argmax(predictions)
        
        # Define class labels
        class_labels = ['Malignant', 'Normal', 'Benign']
        
        # Get the predicted class name
        prediction = class_labels[predicted_class]
        
        # Remove the temporary image file
        os.remove(image_path)
        
        # Return the prediction result to the user
        return render_template('lr_result.html', prediction=prediction, image = image_path)

@app.route('/cbot')
def cbot():
    return render_template('chatbot.html')



if __name__ == '__main__':
    app.run(debug=True)
