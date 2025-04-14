import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Model
import tensorflow as tf

class figureutils:
    """
    A class for figure utilities.
    """

    def __init__(self):
        pass
    # @staticmethod
    def save_figure(self,figure, filename):
        """
        Save a figure to a file.

        Args:
            figure: The figure to save.
            filename: The name of the file to save the figure to.
        """
        figure.savefig(filename)
        print(f"Figure saved as {filename}")


    # Performans metriklerini görselleştirme
    def plot_history(self,history, model_name):
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
        plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
        plt.title(f'{model_name} - Doğruluk')
        plt.legend()
        
        plt.subplot(1,3,2)
        plt.plot(history.history['loss'], label='Eğitim Kaybı')
        plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
        plt.title(f'{model_name} - Kayıp')
        plt.legend()



        plt.subplot(1,3,3)
        plt.plot(history.history['iou'], label='Eğitim IoU')
        plt.plot(history.history['val_iou'], label='Doğrulama IoU')
        plt.title('IoU Metriği')
        plt.legend()

        plt.show()


    def sample_visualization(self, seg_model, X_val, y_val):
        """
        Görselleştirme için rastgele örnekler seçme ve tahmin etme.
        """
        sample_idx = np.random.randint(len(X_val))
        pred_mask = seg_model.predict(X_val[sample_idx][np.newaxis, ..., np.newaxis]).squeeze()

        plt.figure(figsize=(15,5))
        plt.subplot(1,3,1)
        plt.imshow(X_val[sample_idx], cmap='gray')
        plt.title('Orijinal Görüntü')

        plt.subplot(1,3,2)
        plt.imshow(y_val[sample_idx], cmap='gray')
        plt.title('Gerçek Maske')

        plt.subplot(1,3,3)
        plt.imshow(pred_mask > 0.5, cmap='gray')
        plt.title('Tahmin Edilen Maske')
        plt.show()

    def make_gradcam_heatmap(self,img_array, model, last_conv_layer_name, pred_index=None):
        grad_model = Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        
        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap_np=  heatmap.numpy()
            
        # Görselleştirme
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.imshow(img_array[0])
        plt.title('Original')
        plt.subplot(1,2,2)
        plt.imshow(img_array[0])
        plt.imshow(heatmap_np, cmap='jet', alpha=0.5)
        plt.title('Grad-CAM')
        plt.show()