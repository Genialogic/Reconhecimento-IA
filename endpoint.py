from flask import Flask, request, jsonify
import os
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Caminho onde o modelo padrão é salvo
model_path = 'models/model.keras'

# Pasta onde as imagens serão salvas
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Rótulos das classes
class_names = ['Gato', 'Cachorro']

@app.route('/recognize', methods=['POST'])
def recognize():
    if 'file' not in request.files:
        return jsonify({"message": "Não foi encontrado nenhum arquivo na requisição."}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"message": "Nenhum arquivo foi selecionado."}), 400
    
    if file:
        timestampName = str(time.time()).replace('.', '')
        filepath = os.path.join(UPLOAD_FOLDER, timestampName+'.jpg')
        file.save(filepath)

        # Carregar modelo
        model = tf.keras.models.load_model(model_path)    

        # Lê o arquivo e redimensiona a imagem
        img = image.load_img(filepath, target_size=(160,160))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array).flatten()
        prediction = tf.where(predictions < 0.5, 0, 1)

        # Apagar a foto
        os.remove(filepath)

        return jsonify({"message": class_names[prediction.numpy()[0]]}), 200


if __name__ == '__main__':
    app.run(debug=True)
