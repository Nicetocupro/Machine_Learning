import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class TyphoonPCAVisualizer:
    def __init__(self, file_path, n_components=2):
        """
        初始化 TyphoonPCAVisualizer 类
        :param file_path: 数据文件路径
        :param n_components: PCA 的主成分数量
        """
        self.file_path = file_path
        self.n_components = n_components
        self.data = None
        self.pca = PCA(n_components=n_components)
        self.scale = StandardScaler()
        self.x_transformed = None

    def load_and_clean_data(self):
        """
        加载并清洗数据，完成数据预处理
        """
        try:
            # 读取 CSV 数据
            self.data = pd.read_csv(self.file_path, index_col=0)
            
            # 经纬度缩放
            self.data[["Latitude of the center", "Longitude of the center"]] /= 10
            
            # 移除无关列
            self.data.drop(["Indicator of landfall or passage"], axis=1, inplace=True)
            
            # 仅保留1977年及以后的数据
            self.data = self.data[self.data["year"] >= 1977]
            
            # 移除特定的台风等级
            idx = self.data["grade"] == "Just entering into the responsible area of RSMC Tokyo-Typhoon Center"
            self.data = self.data.drop(self.data[idx].index)
            
            # 替换台风等级缩写
            self.data = self.data.replace({
                "Tropical Depression": "TD",
                "Severe Tropical Storm": "STS",
                "Tropical Storm": "TS",
                "Extra-tropical Cyclone": "L",
                "Typhoon": "TY"
            })
            
            # 处理缺失值
            self.data = self.data.dropna()
            
            # 删除额外的列
            self.data = self.data.drop(["Direction of the longest radius of 50kt winds or greater", 
                                        "Direction of the longest radius of 30kt winds or greater"], axis=1)
            
            # 将等级映射到数值
            grade_map = {"TD": 1, "TS": 2, "STS": 3, "TY": 4}
            self.data["grade"] = self.data["grade"].map(grade_map)
            
            # 仅保留每6小时的数据
            self.data = self.data[self.data["hour"] % 6 == 0]

        except Exception as e:
            print(f"Error loading data: {e}")

    def preprocess_data(self):
        """
        预处理数据：标准化并进行 PCA 降维
        """
        try:
            # 提取特征数据（除去 'grade' 列）
            features = self.data.drop(columns=["grade"])
            
            # 标准化特征数据
            x_scaled = self.scale.fit_transform(features)
            
            # 进行 PCA 降维
            self.x_transformed = self.pca.fit_transform(x_scaled)
            
            print(f"Explained variance ratio: {self.pca.explained_variance_ratio_}")
            
            # 返回标签（台风等级）
            return self.data["grade"].values

        except Exception as e:
            print(f"Error during preprocessing: {e}")

    def plot_pca(self, labels):
        """
        绘制 PCA 降维后的数据
        :param labels: 台风等级标签
        """
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(self.x_transformed[:, 0], self.x_transformed[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Grade')
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("Typhoon Data PCA Visualization")
        plt.show()

    def visualize(self):
        """
        主流程：加载数据、预处理数据并进行可视化
        """
        self.load_and_clean_data()
        labels = self.preprocess_data()
        self.plot_pca(labels)

if __name__ == "__main__":
    try:
        # 创建 TyphoonPCAVisualizer 对象，并调用可视化方法
        visualizer = TyphoonPCAVisualizer("../data/archive/typhoon_data.csv")
        visualizer.visualize()
    except Exception as e:
        print(f"An error occurred: {e}")
