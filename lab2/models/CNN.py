import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class SimpleCNN:
    def __init__(self, input_shape, num_classes=1, learning_rate=0.001):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.build_model()

    def build_model(self):
        # 定义一个使用 Keras 构建的简单 CNN 模型
        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=self.input_shape),
            tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.num_classes, activation='sigmoid')  # 使用 sigmoid 进行二分类
        ])
        # 编译模型
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

    def train(self, X_train, y_train, epochs=10, batch_size=32, validation_split=0.2):
        # 确保输入是 numpy 数组
        X_train = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
        y_train = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train

        # 调整输入形状为 CNN 所需的形状 (样本数, 特征数, 通道数)
        X_train = X_train[:, :, np.newaxis]

        # 训练模型
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    def predict(self, X):
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X

        # 调整输入形状为 CNN 所需的形状 (样本数, 特征数, 通道数)
        X = X[:, :, np.newaxis]

        y_pred_proba = self.model.predict(X)
        return (y_pred_proba > 0.5).astype(int)

    def evaluate(self, X_test, y_test):
        # 确保输入是 numpy 数组
        X_test = X_test.to_numpy() if isinstance(X_test, pd.DataFrame) else X_test
        y_test = y_test.to_numpy() if isinstance(y_test, pd.Series) else y_test

        # 调整输入形状为 CNN 所需的形状 (样本数, 特征数, 通道数)
        X_test = X_test[:, :, np.newaxis]

        # 获取预测的概率值
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # 计算 AUC 和 AP
        auc_score = roc_auc_score(y_test, y_pred_proba)
        ap_score = average_precision_score(y_test, y_pred_proba)

        # 打印分类报告
        print(classification_report(y_test, y_pred))

        # 生成混淆矩阵并显示
        #cm = confusion_matrix(y_test, y_pred)
        #ConfusionMatrixDisplay(cm).plot()
        #plt.show()

        # 打印 AUC 和 AP
        print(f"AUC: {auc_score:.4f}")
        print(f"Average Precision (AP): {ap_score:.4f}")
