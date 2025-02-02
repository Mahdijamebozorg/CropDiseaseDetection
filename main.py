from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import urllib

from fastapi import FastAPI
from fastapi import File, UploadFile

model_dir = "./model/efficientnetb3-PlantVillageDisease.h5"
weights_dir = './model/efficientnetb3-PlantVillageDisease-weights.h5'
model_url = 'https://github.com/Mahdijamebozorg/CropDiseaseDetection/raw/main/model/efficientnetb3-PlantVillageDisease.h5'
weights_url = 'https://github.com/Mahdijamebozorg/CropDiseaseDetection/raw/main/model/efficientnetb3-PlantVillageDisease-weights.h5'

model = ...

try:
    model = load_model(model_dir)
except:
    urllib.request.urlretrieve(model_url, model_dir)
    try:
        model = load_model(model_dir)
    except:
        raise RuntimeError("Can't load model")


try:
    model.load_weights(weights_dir)
except:
    urllib.request.urlretrieve(weights_url, weights_dir)
    try:
        model.load_weights(weights_dir)
    except:
        raise RuntimeError("Can't load weights")


# added manually to avoid adding pandas
model_classes = [
    "Apple___Apple_scab", 
    "Apple___Black_rot", 
    "Apple___Cedar_apple_rust", 
    "Apple___healthy", 
    "Blueberry___healthy", 
    "Cherry_(including_sour)___Powdery_mildew", 
    "Cherry_(including_sour)___healthy", 
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", 
    "Corn_(maize)___Common_rust_", 
    "Corn_(maize)___Northern_Leaf_Blight", 
    "Corn_(maize)___healthy", 
    "Grape___Black_rot", 
    "Grape___Esca_(Black_Measles)", 
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", 
    "Grape___healthy", 
    "Orange___Haunglongbing_(Citrus_greening)", 
    "Peach___Bacterial_spot", 
    "Peach___healthy", 
    "Pepper,_bell___Bacterial_spot", 
    "Pepper,_bell___healthy", 
    "Potato___Early_blight", 
    "Potato___Late_blight", 
    "Potato___healthy", 
    "Raspberry___healthy", 
    "Soybean___healthy", 
    "Squash___Powdery_mildew", 
    "Strawberry___Leaf_scorch", 
    "Strawberry___healthy", 
    "Tomato___Bacterial_spot", 
    "Tomato___Early_blight", 
    "Tomato___Late_blight", 
    "Tomato___Leaf_Mold", 
    "Tomato___Septoria_leaf_spot", 
    "Tomato___Spider_mites Two-spotted_spider_mite", 
    "Tomato___Target_Spot", 
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", 
    "Tomato___Tomato_mosaic_virus", 
    "Tomato___healthy"
]


app = FastAPI()

@app.get("/")
async def root():
    return {
        "name": "CropDiseaseAPI",
        "description": "An API for crop disease detection model",
        "version": "1.0.0",
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        img = img.resize((224,224))
        # convert to array
        img = tf.keras.utils.img_to_array(img)
        if img is None:
            return {"message": "Image file is not valid"}

        predictions = model.predict(tf.convert_to_tensor(tf.expand_dims(img,axis=0)))
        prediction_class = model_classes[tf.argmax(predictions,axis=1).numpy()[0]]
        return {"prediction_class": prediction_class}
    except Exception as e:
        return {"message": f"{repr(e)}"}
    finally:
        file.file.close()