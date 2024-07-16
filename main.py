import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Pasta para armazenar as imagens específicas para utilização do modelo
images_folder = 'specific_images'

# Caminho onde o modelo padrão é salvo
model_path = 'models/model.keras'

# Pastas para armazenar os datasets de treino e validação
dataset_dir = 'dataset'
dataset_train_dir = os.path.join(dataset_dir, 'train')
dataset_validation_dir = os.path.join(dataset_dir, 'validation')

# Configurações do modelo
image_width, image_height = 160, 160
image_color_channel = 3
image_color_channel_size = 255
image_size = (image_width, image_height)
image_shape = image_size + (image_color_channel,)

# Configurações do treinamento
batch_size = 32
epochs = 20
learning_rate = 0.0001

# Rótulos das classes
class_names = ['Gato', 'Cachorro']

# Carregar datasets
dataset_train = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_train_dir,
    image_size=image_size,
    batch_size=batch_size,
    shuffle=True
)
dataset_validation = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_validation_dir,
    image_size=image_size,
    batch_size=batch_size,
    shuffle=True
)

# Dividir o dataset de validação em 5 partes
dataset_validation_cardinality = tf.data.experimental.cardinality(dataset_validation)
dataset_validation_batches = dataset_validation_cardinality // 5
dataset_validation = dataset_validation.skip(dataset_validation_batches)

def clearTerminal():
    os.system('clear' if os.name == 'posix' else 'cls') 

while True:
    clearTerminal()
    print("1 - Carregar modelo treinado e plotar testes.")
    print("2 - Treinar e criar um novo modelo.")
    escolha = input('')

    if escolha == '1' or escolha == '2':
        break;

if escolha == '1':
    # Carregar modelo
    model = tf.keras.models.load_model(model_path)            

    # Plotar testes
    def plot_specific_images_predictions(model, image_size=(160, 160)):
        plt.gcf().clear()
        plt.figure(figsize=(15, 15))
        
        # Lista todos os arquivos na pasta
        image_files = [os.path.join(images_folder, f) for f in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, f))]
        
        for i, img_path in enumerate(image_files[:9]):  # Limita a 9 imagens para manter o layout 3x3
            img = image.load_img(img_path, target_size=image_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Cria um batch com uma única imagem
            
            predictions = model.predict(img_array).flatten()
            prediction = tf.where(predictions < 0.5, 0, 1)
            
            plt.subplot(3, 3, i + 1)
            plt.axis('off')
            plt.imshow(image.load_img(img_path))
            plt.title(class_names[prediction.numpy()[0]])
        
        plt.show()

    plot_specific_images_predictions(model)

elif escolha == '2':
    # Definir o modelo
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=image_shape),
        tf.keras.layers.Rescaling(1./image_color_channel_size),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compilar o modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )

    # Treinar o modelo
    history = model.fit(
        dataset_train,
        validation_data=dataset_validation,
        epochs=epochs
    )

    # Salvar o modelo
    model.save(model_path)

    clearTerminal()
    print("Modelo treinado e salvo com sucesso!")