import numpy as np
import tensorflow as tf
import json
import joblib
from keras.models import load_model
from keras.applications.vgg19 import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
import requests

model = load_model('VGG19_test_model.h5')

def download_image(url, filename):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for any HTTP errors
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f"Image downloaded successfully as {filename}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

def process_image(path):
    img = tf.keras.utils.load_img(path, target_size=(256,256))
    i = tf.keras.utils.img_to_array(img)
    ims = preprocess_input(i)
    imgs = np.expand_dims(ims, axis=0)
    return imgs
    

def classify(path):
    
    classls=  ["Alpinia Galanga (Rasna)",
    "Amaranthus Viridis (Arive-Dantu)",
    "Artocarpus Heterophyllus (Jackfruit)",
    "Azadirachta Indica (Neem)",
    "Basella Alba (Basale)",
    "Brassica Juncea (Indian Mustard)",
    "Carissa Carandas (Karanda)",
    "Citrus Limon (Lemon)",
    "Ficus Auriculata (Roxburgh fig)",
    "Ficus Religiosa (Peepal Tree)",
    "Hibiscus Rosa-sinensis",
    "Jasminum (Jasmine)",
    "Mangifera Indica (Mango)",
    "Mentha (Mint)",
    "Moringa Oleifera (Drumstick)",
    "Muntingia Calabura (Jamaica Cherry-Gasagase)",
    "Murraya Koenigii (Curry)",
    "Nerium Oleander (Oleander)",
    "Nyctanthes Arbor-tristis (Parijata)",
    "Ocimum Tenuiflorum (Tulsi)",
    "Piper Betle (Betel)",
    "Plectranthus Amboinicus (Mexican Mint)",
    "Pongamia Pinnata (Indian Beech)",
    "Psidium Guajava (Guava)",
    "Punica Granatum (Pomegranate)",
    "Santalum Album (Sandalwood)",
    "Syzygium Cumini (Jamun)",
    "Syzygium Jambos (Rose Apple)",
    "Tabernaemontana Divaricata (Crape Jasmine)",
    "Trigonella Foenum-graecum (Fenugreek)"]
    
    img = process_image(path)
    result = model.predict(img)
    index = np.argmax(result)
    return classls[index]

if __name__ == "__main__":
    print(classify("E:\\My Projects\\Medicinal_plant_detection\\test\\Muntingia Calabura (Jamaica Cherry-Gasagase)\\MC-S-038.jpg"))

    