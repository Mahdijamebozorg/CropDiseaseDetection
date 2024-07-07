import numpy as np
import cv2
import pandas as pd
from keras.models import load_model 

from fastapi import FastAPI
from fastapi import File, UploadFile

model_dir = "./model/efficientnetb3-Plant Village Disease-99.65.h5"
weights_dir = './model/efficientnetb3-Plant Village Disease-weights.h5'
classes_dir = './class_dict.csv'

model = load_model(model_dir)
model.load_weights(weights_dir)
model_classes = pd.read_csv(classes_dir)['class']


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
        nparr = np.fromstring(contents, np.uint8)
        img = cv2.resize(cv2.imdecode(nparr, cv2.IMREAD_COLOR),(224,224))
        if img is None:
            return {"message": "Image file is not valid"}

        predictions = model.predict(np.array([img]))
        prediction = np.argmax(predictions,axis=1)
        prediction_class = model_classes[prediction]
        return {"prediction_class": prediction_class}
    except Exception as e:
        return {"message": f"{repr(e)}"}
    finally:
        file.file.close()



