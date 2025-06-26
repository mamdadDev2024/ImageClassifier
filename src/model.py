import os
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import EarlyStopping

class Classifier:
    def __init__(self, data_dir='data', image_size=(128, 128), batch_size=32, epochs=10):
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.epochs = epochs

        self._remove_broken_images()
        self._load_datasets()
        self._build_model()

    def _remove_broken_images(self):
        """حذف تصاویر خراب برای جلوگیری از ارور در زمان بارگیری"""
        valid_exts = ('.jpg', '.jpeg', '.png')
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.lower().endswith(valid_exts):
                    path = os.path.join(root, file)
                    try:
                        img = tf.io.read_file(path)
                        _ = tf.image.decode_image(img, channels=3).shape
                    except Exception:
                        print(f"تصویر خراب حذف شد: {path}")
                        os.remove(path)

    def _load_datasets(self):
        """بارگیری داده‌ها و نرمال‌سازی تصاویر"""
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset='training',
            seed=42,
            image_size=self.image_size,
            batch_size=self.batch_size
        )

        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset='validation',
            seed=42,
            image_size=self.image_size,
            batch_size=self.batch_size
        )

        # نرمال‌سازی داده ها
        self.train_ds = self.train_ds.map(lambda x, y: (x / 255.0, y))
        self.val_ds = self.val_ds.map(lambda x, y: (x / 255.0, y))

    def _build_model(self):
        """ساخت مدل CNN"""
        self.model = Sequential([
            Input(shape=(*self.image_size, 3)),
            Conv2D(16, 3, activation='relu'),
            MaxPooling2D(),
            Conv2D(32, 3, activation='relu'),
            MaxPooling2D(),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')  # برای binary classification
        ])

        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

        self.model.summary()

    def train(self):
        """آموزش مدل با early stopping"""
        early_stop = EarlyStopping(patience=3, restore_best_weights=True)
        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.epochs,
            callbacks=[early_stop]
        )

    def plot_metrics(self):
        """نمایش نمودارهای عملکرد مدل"""
        if not hasattr(self, 'history'):
            print("ابتدا متد train را اجرا کنید.")
            return

        history = self.history.history

        plt.figure()
        plt.plot(history['accuracy'], label='Train Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(history['loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def save_model(self, path='model.keras'):
        self.model.save(path)
        print(f"saved model {path}")
