import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, optimizers
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense
from sklearn.mixture import GaussianMixture
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers.schedules import ExponentialDecay

class DEC:
    def __init__(self, n_clusters, alpha=1.0, initial_lr=0.001, decay_steps=10000, decay_rate=0.96):
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.autoencoder = None
        self.encoder = None
        self.cluster_centers = None

        # 定义学习率调度器
        self.learning_rate_schedule = ExponentialDecay(
            initial_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True  # 每过 decay_steps 才衰减一次
        )
    def build_autoencoder(self, input_dim):
        input_layer = layers.Input(shape=(input_dim,))

        # 编码器部分
        encoded = layers.Dense(512, activation='leaky_relu')(input_layer)
        encoded = layers.BatchNormalization()(encoded)
        encoded = layers.Dropout(0.1)(encoded)

        encoded = layers.Dense(256, activation='leaky_relu')(encoded)
        encoded = layers.BatchNormalization()(encoded)
        encoded = layers.Dropout(0.1)(encoded)

        encoded = layers.Dense(128, activation='leaky_relu')(encoded)
        encoded = layers.BatchNormalization()(encoded)

        code = layers.Dense(16, activation='linear')(encoded)

        # 解码器部分
        decoded = layers.Dense(128, activation='leaky_relu')(code)
        decoded = layers.BatchNormalization()(decoded)

        decoded = layers.Dense(256, activation='leaky_relu')(decoded)
        decoded = layers.BatchNormalization()(decoded)

        decoded = layers.Dense(512, activation='leaky_relu')(decoded)
        decoded = layers.BatchNormalization()(decoded)

        decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)

        self.autoencoder = models.Model(input_layer, decoded)
        self.encoder = models.Model(input_layer, code)

        optimizer = optimizers.Adam(learning_rate=self.learning_rate_schedule)
        self.autoencoder.compile(optimizer=optimizer, loss='mse')

    def train_autoencoder(self, X, epochs=100, batch_size=256):
        print("Training autoencoder...")
        self.autoencoder.fit(X, X, epochs=epochs, batch_size=batch_size, verbose=1)
        print("Autoencoder training completed.")

    def clustering(self, X, batch_size=256, epochs=100):
        encoded_X = self.encoder.predict(X, batch_size=batch_size)
        print("Encoded X shape in clustering:", encoded_X.shape)

        #kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        #kmeans.fit(encoded_X)
        #self.cluster_centers = kmeans.cluster_centers_

        # 替换KMeans为GMM
        gmm = GaussianMixture(n_components=self.n_clusters, random_state=42)
        gmm.fit(encoded_X)
        self.cluster_centers = gmm.means_

        prev_labels = np.zeros(X.shape[0])
        for epoch in range(epochs):
            q = self._calculate_cluster_probabilities(X)
            new_labels = np.argmax(q, axis=1)

            if np.all(new_labels == prev_labels):
                print(f"Converged at epoch {epoch}")
                break

            prev_labels = new_labels
            self._train_autoencoder_with_clusters(X, q)

    def _calculate_cluster_probabilities(self, X):
        encoded_X = self.encoder.predict(X)
        distance = np.linalg.norm(encoded_X[:, np.newaxis] - self.cluster_centers, axis=-1)
        distance = np.exp(-distance ** 2 / (2 * np.std(distance) ** 2))
        q = distance / np.sum(distance, axis=1, keepdims=True)
        return q

    def _train_autoencoder_with_clusters(self, X, q):
        p = (q ** self.alpha) / np.sum(q ** self.alpha, axis=1, keepdims=True)
        loss = np.sum(p * np.log(p / q))

        with tf.GradientTape() as tape:
            reconstruction = self.autoencoder(X, training=True)
            reconstruction_loss = tf.reduce_mean(tf.square(X - reconstruction))

            total_loss = reconstruction_loss + loss

        grads = tape.gradient(total_loss, self.autoencoder.trainable_variables)
        self.autoencoder.optimizer.apply_gradients(zip(grads, self.autoencoder.trainable_variables))
