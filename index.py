import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Definição dos parâmetros iniciais
data_dir = "./data"
img_height, img_width = 150, 150
batch_size = 32
num_classes = 2
epochs = 25

# Preparação dos geradores de dados de treinamento e validação
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,  # 20% dos dados para validação
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',  # 'categorical' para mais de 2 classes
    subset='training',
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',  # 'categorical' para mais de 2 classes
    subset='validation',
)

# Construção do modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid'),  # 'softmax' para mais de 2 classes
])

# Compilação do modelo
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # 'categorical_crossentropy' para mais de 2 classes
              metrics=['accuracy'])

# Treinamento do modelo
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
)

# Avaliação do modelo (opcional)
model.evaluate(validation_generator)

# Salvar o modelo treinado
model.save('model.h5')