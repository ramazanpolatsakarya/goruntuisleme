from dicomutils import  dicomutils
import numpy as np

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf

class datautils:
    def __init__(self):
        pass

    def load_data(self, target_size=(256, 256)):


        utils = dicomutils()

        kanama_path = "Kanama Veri Seti/DICOM"
        iskemi_path = "İskemi Veri Seti/DICOM"
        normal_path = "İnme Yok Veri Seti/DICOM"

        print("Veri yükleniyor...")

        kanama_images, kanama_names = utils.load_dicom_data(kanama_path)
        iskemi_images, iskemi_names = utils.load_dicom_data(iskemi_path)
        normal_images, normal_names = utils.load_dicom_data(normal_path)

        # Etiketler (0: kanama, 1: iskemi, 2: normal)
        kanama_labels = np.zeros(len(kanama_images))
        iskemi_labels = np.ones(len(iskemi_images))
        normal_labels = np.full(len(normal_images), 2)



        plt.figure(figsize=(10,4))
        plt.subplot(131)
        plt.imshow(kanama_images[0], cmap='gray')
        plt.title('Kanama')
        plt.subplot(132)
        plt.imshow(iskemi_images[0], cmap='gray')
        plt.title('İskemi')
        plt.subplot(133)
        plt.imshow(normal_images[0], cmap='gray')
        plt.title('Normal')
        plt.show()


        print(f"Kanama veri şekli: {kanama_images.shape}")
        print(f"İskemi veri şekli: {iskemi_images.shape}") 
        print(f"Normal veri şekli: {normal_images.shape}")


        print(f"Kanama etiketleri: {kanama_labels.shape}")
        print(f"İskemi etiketleri: {iskemi_labels.shape}") 
        print(f"Normal etiketleri: {normal_labels.shape}")


        # Tüm verileri ve etiketleri birleştirme
        all_images = np.concatenate([kanama_images, iskemi_images, normal_images])
        all_labels = np.concatenate([kanama_labels, iskemi_labels, normal_labels])

        # Görüntü boyutlarını kontrol et ve normalize et
        print("Görüntü şekilleri:", [img.shape for img in all_images[:5]])



        # Boyutları standartlaştırma (örneğin 256x256)
        resized_images = []
        for img in all_images:
            # Burada basit bir boyutlandırma yapıyoruz, daha iyi yöntemler kullanılabilir
            img_normalized = (img - np.min(img)) / (np.max(img) - np.min(img))  # 0-1 arası normalize
            img_resized = tf.image.resize(img_normalized[..., np.newaxis], target_size)
            resized_images.append(img_resized.numpy().squeeze())

        X = np.array(resized_images)
        y = all_labels



        # Veriyi eğitim ve test olarak ayırma
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        # Veri artırma (data augmentation) pipeline'ı
        # data_augmentation = tf.keras.Sequential([
        #     layers.RandomFlip("horizontal"),
        #     layers.RandomRotation(0.1),
        #     layers.RandomZoom(0.1),
        # ])


        # Görüntüleri 3 kanala çoğaltma (RGB formatı için)
        X_train_rgb = np.repeat(X_train[..., np.newaxis], 3, axis=-1)
        X_val_rgb = np.repeat(X_val[..., np.newaxis], 3, axis=-1)

        return X_train_rgb, y_train, X_val_rgb, y_val