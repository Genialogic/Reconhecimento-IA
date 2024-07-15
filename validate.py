import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Carregar o modelo treinado
model = load_model('model.h5')

# Mapeamento dos índices de saída para as classes
classes = {0: 'Chihuahua', 1: 'Flores', 2: 'Nenhuma das anteriores'}

# Carregar e pré-processar a imagem de teste
img_path = 'test.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Transforma a imagem em um batch de tamanho 1
img_array /= 255.0  # Normaliza a imagem

# Fazer a previsão
predictions = model.predict(img_array)
predicted_class_index = np.argmax(predictions, axis=1)[0]
predicted_class = classes[predicted_class_index]
probabilidade = predictions[0][predicted_class_index]

# Interpretar a saída
print(f"Classe provável: {predicted_class}, Probabilidade: {probabilidade:.2f}")