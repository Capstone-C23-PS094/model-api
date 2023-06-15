import io

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from pydantic import BaseModel
from fastapi import FastAPI, Form, UploadFile
from typing import Annotated

# Tensorflow
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_file
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
import tensorflow_decision_forests
import numpy as np

from data import crop_data, leaf_disease
from gcs import GCStorage

app = FastAPI()

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=methods,
    allow_headers=headers,
)


@app.get("/")
async def root():
    return {"message": "Welcome !"}


class CropInput(BaseModel):
    nitrogen: int = 0
    phosphorous: int = 0
    potassium: int = 0
    temperature: int = 0
    humidity: int = 0
    ph: int = 0
    rainfall: int = 0


@app.post("/crop-recommendations")
async def get_crop_recommendations(
        nitrogen: Annotated[float, Form()],
        phosphorous: Annotated[float, Form()],
        potassium: Annotated[float, Form()],
        temperature: Annotated[float, Form()],
        humidity: Annotated[float, Form()],
        ph: Annotated[float, Form()],
        rainfall: Annotated[float, Form()]
):
    model = tf.saved_model.load('./model/crop_recommdation')

    input_data = {
        "N": np.array([nitrogen]),
        "P": np.array([phosphorous]),
        "K": np.array([potassium]),
        "temperature": np.array([temperature]),
        "humidity": np.array([humidity]),
        "ph": np.array([ph]),
        "rainfall": np.array([rainfall])
    }

    pred = model.serve(input_data)

    # Daftar nama kelas
    class_names = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans', 'mungbean', 'blackgram',
                   'lentil', 'pomegranate', 'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple',
                   'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee']

    # Mencari indeks dengan nilai prediksi tertinggi

    predicted_label = pred[0]
    predicted_index = np.argmax(predicted_label)

    # Mengambil nama kelas yang sesuai dengan indeks
    predicted_class = class_names[predicted_index]

    return crop_data[predicted_class]


def get_predict_detail(predictions, classes):
    predicts = []
    class_names = ['bacterial_leaf_blight', 'brown_spot', 'healthy', 'leaf_blast', 'leaf_scald', 'narrow_brown_spot']
    print(predictions)
    for x in predictions.tolist():
        print(x)
        predicts.append({
            "disease": leaf_disease[class_names[x]],
            "percentage": classes[x],
        })
    return predicts


@app.post("/rice-disease-detection")
async def get_rice_leaf_disease_detection(image: UploadFile):
    temp_image = image
    image_link = GCStorage().upload_file(image)

    model_dir = "./model/rice_leaf_detection.h5"
    model = load_model(model_dir)

    contents = temp_image.file.read()
    temp_file = io.BytesIO()
    temp_file.write(contents)
    temp_file.seek(0)
    load_image = load_img(temp_file, target_size=(224, 224))
    x = img_to_array(load_image)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    classes = model.predict(images)[0]
    prediction_probabilities = tf.math.top_k(classes, k=3)

    predictions = prediction_probabilities.indices.numpy()

    return {
        "predictions": get_predict_detail(predictions, classes.tolist()),
        "image_url": image_link,
    }


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    uvicorn.run(app, host="0.0.0.0", port=8000)
