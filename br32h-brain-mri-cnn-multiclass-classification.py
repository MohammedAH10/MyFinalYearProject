
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_dir = "/kaggle/input/brain-tumor-mri-dataset/Training"

test_dir = "/kaggle/input/brain-tumor-mri-dataset/Testing"

train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            image_size=(224, 224),
                                                            batch_size=32,
                                                            label_mode='int')

test_dataset = tf.keras.utils.image_dataset_from_directory(test_dir,
                                                            image_size=(224, 224),
                                                            batch_size=32,
                                                            label_mode='int')


class_names = train_dataset.class_names
class_names

# Visualizing the data
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns   

sns.set()
sns.set_context('notebook')


for images, labels in train_dataset.take(1):
    for i in range(5):
        plt.figure(figsize=(6, 6))
        np_image = images[i].numpy().astype("uint8")
        print(f"LABEL:{class_names[labels[i]]}")
        print(f"\nIMAGE PIXEL ARRAY:\n\n{np_image}\n\n")

        plt.imshow(np_image)
        plt.colorbar()
        plt.show()

normalization_layer = tf.keras.layers.Rescaling(1./255)

train_dataset = train_dataset.map(lambda x, y:(normalization_layer(x), y))
test_dataset = test_dataset.map(lambda x, y:(normalization_layer(x), y))

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)


model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(4, activation='softmax')
    
    ])


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])


maivas = model.fit(train_dataset,
                    validation_data=test_dataset,
                    epochs=10)



test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test Accuracy:{test_acc:.2f}")# Plot accuracy
plt.plot(maivas.history['accuracy'], label='Training Accuracy')
plt.plot(maivas.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

# Plot loss
plt.plot(maivas.history['loss'], label='Training Loss')
plt.plot(maivas.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

model.save('brain_mri.h5')