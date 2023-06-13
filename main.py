from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from pydantic import BaseModel

# Tensorflow
import tensorflow as tf
import tensorflow_decision_forests
import numpy as np

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
async def get_crop_recommendations(crop_in: CropInput):
    model = tf.saved_model.load('./model/crop_recommdation')

    input_data = {
        "N": np.array([crop_in.nitrogen]),
        "P": np.array([crop_in.phosphorous]),
        "K": np.array([crop_in.potassium]),
        "temperature": np.array([crop_in.temperature]),
        "humidity": np.array([crop_in.humidity]),
        "ph": np.array([crop_in.ph]),
        "rainfall": np.array([crop_in.rainfall])
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

    return {
        "model-prediction": predicted_class
    }


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    uvicorn.run(app, host="0.0.0.0", port=8000)
