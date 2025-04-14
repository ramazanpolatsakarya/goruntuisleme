

import pydicom
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='skimage')
from tensorflow.keras import layers, models
from PIL import Image

from tensorflow.keras.applications import ResNet50, EfficientNetB4
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import layers, models

class dicomutils:
    """
    A class that provides utility functions for working with DICOM files.
    """
    def __init__(self):
        pass
    # @staticmethod
    def get_dicom_tags(self,dicom_file):
        """
        Extracts DICOM tags from a DICOM file.

        Args:
            dicom_file (str): Path to the DICOM file.

        Returns:
            dict: A dictionary containing DICOM tags and their values.
        """

        # Read the DICOM file
        ds = pydicom.dcmread(dicom_file)

        # Extract tags and their values
        tags = {tag: ds.get(tag) for tag in ds.keys()}

        return tags
    # 2. Model oluşturma
    # @staticmethod
    def build_classification_model(self,input_shape, num_classes):
        model = models.Sequential([
            layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(64, (3,3), activation='relu'),
            layers.MaxPooling2D((2,2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        return model


    # @staticmethod
    def build_segmentation_model(self,input_shape=(256, 256, 3)):
        inputs = tf.keras.Input(shape=input_shape)
        
        # Encoder
        x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D()(x)
        
        # Bottleneck
        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        
        # Decoder
        x = layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')(x)
        x = layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')(x)
        
        # Output (2 sınıf: lezyon/arkaplan)
        outputs = layers.Conv2D(1, 1, activation='sigmoid')(x)
        
        seg_model=  tf.keras.Model(inputs, outputs)
    

        seg_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                    loss='binary_crossentropy',
                    metrics=['accuracy', tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1])])
        
        return seg_model


    def segmentation_model_fit(self, X_train_rgb, y_train, X_val_rgb, y_val):
        seg_model = self.build_segmentation_model()

        # Callback'ler
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
            tf.keras.callbacks.ModelCheckpoint('best_seg_model.h5', save_best_only=True)
        ]

        # Modeli eğitme
        history = seg_model.fit(
            X_train_rgb[..., np.newaxis], y_train[..., np.newaxis],
            validation_data=(X_val_rgb[..., np.newaxis], y_val[..., np.newaxis]),
            epochs=50,
            batch_size=8,
            callbacks=callbacks
        )

        return seg_model, history


    # @staticmethod
    def build_resnet_model(self,input_shape=(256, 256, 3), num_classes=3):
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        
        # Katmanları dondur
        for layer in base_model.layers:
            layer.trainable = False
            
        # Özellik çıkarma
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        
        resnet_model= Model(inputs=base_model.input, outputs=predictions)
        resnet_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return resnet_model

    # @staticmethod
    def fit_resnet_model(self, X_train_rgb, y_train, X_val_rgb, y_val):
        # ResNet Modeli
        resnet_model = self.build_resnet_model()


        # Callback'ler
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
            tf.keras.callbacks.ModelCheckpoint('best_model_ResNet50.h5', save_best_only=True)
        ]

        # ResNet eğitimi
        print("ResNet50 eğitiliyor...")
        resnet_history = resnet_model.fit(
            X_train_rgb, y_train,
            validation_data=(X_val_rgb, y_val),
            epochs=20,
            batch_size=16,
            callbacks=callbacks
        )

        return resnet_model,resnet_history


    # @staticmethod
    def build_efficientnet_model(self,input_shape=(256, 256, 3), num_classes=3, learning_rate=0.001, dense_units=1024):
        base_model = EfficientNetB4(weights='imagenet', include_top=False, input_shape=input_shape)
        
        # Katmanları dondur
        for layer in base_model.layers:
            layer.trainable = False
            
        # Özellik çıkarma
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(dense_units, activation='relu')(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        
        effnet_model= Model(inputs=base_model.input, outputs=predictions)
        effnet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return effnet_model


    def fit_efficientnet_model(self, X_train_rgb, y_train, X_val_rgb, y_val):
        # EfficientNet Modeli
        effnet_model = self.build_efficientnet_model()


        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
            tf.keras.callbacks.ModelCheckpoint('best_model_effnet_model.h5', save_best_only=True)
        ]

        # EfficientNet eğitimi
        print("\nEfficientNetB4 eğitiliyor...")
        effnet_history = effnet_model.fit(
            X_train_rgb, y_train,
            validation_data=(X_val_rgb, y_val),
            epochs=20,
            batch_size=16,
            callbacks=callbacks
        )

        return effnet_model,effnet_history
    # @staticmethod
    def load_mask_data(self,dicom_folder, overlay_folder, target_size=(256, 256)):
        dicom_files = [f for f in os.listdir(dicom_folder) if f.endswith('.dcm')]
        images = []
        masks = []
        
        for file in dicom_files:
            try:
                # DICOM görüntüsünü yükle
                ds = pydicom.dcmread(os.path.join(dicom_folder, file))
                img = ds.pixel_array


                # Görüntüyü normalize et (0-1 arası)
                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                
                # Boyutlandırma yap
                if img.shape != target_size:
                    img = tf.image.resize(img[np.newaxis, ..., np.newaxis], target_size)
                    img = img.numpy().squeeze()
                    


                
                # Overlay maskesini yükle (PNG veya DICOM overlay)
                overlay_path = os.path.join(overlay_folder, file.replace('.dcm', '.png'))
                print(overlay_path)
                if os.path.exists(overlay_path):
                    mask = np.array(Image.open(overlay_path).convert('L'))
                    mask = (mask > 0).astype(np.float32)  # Binary maskeye dönüştür
                    
                    # Boyutları eşleştir
                    if img.shape != mask.shape:
                        mask = tf.image.resize(mask[np.newaxis, ..., np.newaxis], img.shape)
                        mask = mask.numpy().squeeze()
                    
                    images.append(img)
                    masks.append(mask)
                    
            except Exception as e:
                print(f"{file} işlenirken hata: {str(e)}")
                continue
                
        return np.array(images), np.array(masks)
    # @staticmethod
    def load_dicom_data(self,folder_path, target_size=(256, 256)):
        """DICOM dosyalarını yükler ve boyutlarını standartlaştırır"""
        dicom_files = [f for f in os.listdir(folder_path) if f.endswith('.dcm')]
        images = []
        names = []
        for file in dicom_files:
            try:
                ds = pydicom.dcmread(os.path.join(folder_path, file))
                img = ds.pixel_array
                
                # Görüntüyü normalize et (0-1 arası)
                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                
                # Boyutlandırma yap
                if img.shape != target_size:
                    img = tf.image.resize(img[np.newaxis, ..., np.newaxis], target_size)
                    img = img.numpy().squeeze()
                    
                images.append(img)
                names.append(file)
            except Exception as e:
                print(f"{file} dosyası işlenirken hata: {str(e)}")
                continue
        print(f"{len(images)} adet görüntü yüklendi.")
        print(f"{len(names)} adet görüntü yüklendi.")

        return np.array(images), np.array(names)