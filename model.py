import os
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import EarlyStopping

class Classifier:
    """_
    """
    
    def __init__(self, data_dir='data', image_size=(256, 256), batch_size=32, epochs=15):
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.clean_broken_images(self.data_dir)
        self.load_data()
        self.build_model()

    def clean_broken_images(self, directory, allowed_ext=["jpg", "jpeg", "png"]):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.split('.')[-1].lower() in allowed_ext:
                    path = os.path.join(root, file)
                    try:
                        img = tf.io.read_file(path)
                        img = tf.image.decode_image(img, channels=3)
                        _ = img.shape
                    except:
                        print(f"crrupted image {path}")
                        os.remove(path)

    def load_data(self):
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset='training',
            seed=123,
            image_size=self.image_size,
            batch_size=self.batch_size
        )

        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset='validation',
            seed=123,
            image_size=self.image_size,
            batch_size=self.batch_size
        )

        self.train_ds = self.train_ds.map(lambda x, y: (x / 255.0, y))
        self.val_ds = self.val_ds.map(lambda x, y: (x / 255.0, y))

    def build_model(self):
        self.model = Sequential([
            Input(shape=(*self.image_size, 3)),
            Conv2D(16, 3, activation='relu'),
            MaxPooling2D(),
            Conv2D(32, 3, activation='relu'),
            MaxPooling2D(),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        self.model.summary()

    def train(self):
        callback = EarlyStopping(patience=3, restore_best_weights=True)
        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.epochs,
            callbacks=[callback]
        )

    def plot_metrics(self):
        if hasattr(self, 'history'):
            plt.figure()
            plt.plot(self.history.history['accuracy'], label='Train Accuracy')
            plt.plot(self.history.history['val_accuracy'], label='Val Accuracy')
            plt.legend()
            plt.title("Accuracy")
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.show()

            plt.figure()
            plt.plot(self.history.history['loss'], label='Train Loss')
            plt.plot(self.history.history['val_loss'], label='Val Loss')
            plt.legend()
            plt.title("Loss")
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.show()
        else:
            print('run this method after train method please!')

    def save_model(self, path='simple_model.keras'):
        self.model.save(path)
        print(f"saved on {path}")
