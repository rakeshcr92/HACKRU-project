import os
import pickle
import random
import numpy as np
import xgboost
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

def load_saved_model(path):
    model = load_model(path, compile=False)
    return model

def predict_pneumonia(path):
    data = load_img(path, target_size=(224, 224, 3))
    data = np.asarray(data).reshape((-1, 224, 224, 3))
    data = data * 1.0 / 255
    pneumonia_model_path = './website/app_models/pneumonia_model.h5'
    predicted = np.round(load_saved_model(pneumonia_model_path).predict(data)[0])[0]
    return predicted

def value_predictor(to_predict_list):
    if len(to_predict_list) == 15:
        model_path = './website/app_models/kidney_model.pkl'
        page = 'kidney'
    elif len(to_predict_list) == 10:
        model_path = './website/app_models/liver_model.pkl'
        page = 'liver'
    elif len(to_predict_list) == 11:
        model_path = './website/app_models/heart_model.pkl'
        page = 'heart'
 
    else:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        pred = model.predict(np.array(to_predict_list).reshape((-1, len(to_predict_list))))

    if page != 'stroke':
        print(pred[0], page)
    return pred[0], page
