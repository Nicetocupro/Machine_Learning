import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
import folium
import models.RidgeRegression as rr  # 使用岭回归
import models.RandomForest as rf  # 使用随机森林
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import KFold

class TyphoonDataProcessor:
    def __init__(self, file_path):
        # 初始化类
        self.file_path = file_path
        self.data = None
        self.dataset = None
        self.test_typhoons = None
        self.x_train = None
        self.x_val = None
        self.y_train = None
        self.y_val = None
        self.model = rf.RandomForest(n_estimators=100, random_state=0)  # 使用随机森林模型
        self.scale = StandardScaler()
        # self.pca = PCA(n_components=26)  # PCA维数设置为n_components（已注释掉）

    def load_data(self):
        # 读取CSV数据，并调整经纬度数据的单位
        self.data = pd.read_csv(self.file_path, index_col=0)
        self.data[["Latitude of the center", "Longitude of the center"]] /= 10  # 缩成一行
        # 移除无关列
        self.data.drop(["Indicator of landfall or passage"], axis=1, inplace=True)
        # 仅保留1977年及以后的数据
        self.data = self.data[self.data["year"] >= 1977]
        # 移除特定条件下的行
        idx = self.data["grade"] == "Just entering into the responsible area of RSMC Tokyo-Typhoon Center"
        self.data = self.data.drop(self.data[idx].index)
        # 替换特定字符串数据为缩写
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
        self.data = self.data.drop(["Direction of the longest radius of 50kt winds or greater", "Direction of the longest radius of 30kt winds or greater"], axis=1)
        # 将等级映射到数值
        grade_map = {"TD": 1, "TS": 2, "STS": 3, "TY": 4}
        self.data["grade"] = self.data["grade"].map(grade_map)
        # 仅保留每6小时的数据
        self.data = self.data[self.data["hour"] % 6 == 0]
        # 根据国际编号分组，并处理为三维数据集
        ids = self.data["International number ID"].unique()
        typhoons = [self.data[self.data["International number ID"] == ID].drop(["International number ID"], axis=1) for ID in ids]

        interval = 1
        self.dataset = np.empty((0, 3, 13))
        self.test_typhoons = np.empty((0, 3, 13))

        for typhoon in typhoons:
            nptyphoon = np.array(typhoon)
            typhoon_year = typhoon.iloc[0]["year"]
            num_iterations = nptyphoon.shape[0] - 3

            for i in range(0, num_iterations, interval):
                single_data = np.expand_dims(nptyphoon[i:i + 3], axis=0)

                if typhoon_year == 2022:
                    self.test_typhoons = np.append(self.test_typhoons, single_data, axis=0)
                else:
                    self.dataset = np.append(self.dataset, single_data, axis=0)

    def split_data(self):
        # 提取特征和标签
        x = self.dataset[:, :2, :]
        y = self.dataset[:, 2, 5:]
        # 数据集划分
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(x, y, test_size=0.3, random_state=42, shuffle=True)
        print(self.x_train.shape)
        print(self.y_train.shape)
        print(self.x_val.shape)
        print(self.y_val.shape)
        # 数据预处理
        self.x_train = self.x_train.reshape(-1, 26)
        self.x_val = self.x_val.reshape(-1, 26)
        self.x_train = self.scale.fit_transform(self.x_train)
        self.x_val = self.scale.transform(self.x_val)

        # 进行PCA降维处理（注释掉）
        # self.x_train = self.pca.fit_transform(self.x_train)
        # self.x_val = self.pca.transform(self.x_val)
        # print("Explained variance ratio (train set):", self.pca.explained_variance_ratio_)

    def train_model(self):
        # 使用5折交叉验证
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rmse_scores = []
        r2_scores = []

        for train_index, val_index in kf.split(self.x_train):
            x_train_fold, x_val_fold = self.x_train[train_index], self.x_train[val_index]
            y_train_fold, y_val_fold = self.y_train[train_index], self.y_train[val_index]

            # 训练模型
            self.model.train(x_train_fold, y_train_fold)
            pred_val = self.model.predict(x_val_fold)

            # 计算 RMSE 和 R²
            rmse_fold = mean_squared_error(y_val_fold, pred_val, squared=False)
            r2_fold = r2_score(y_val_fold, pred_val)

            rmse_scores.append(rmse_fold)
            r2_scores.append(r2_fold)

        print(f"Cross-validated RMSE: {np.mean(rmse_scores)} ± {np.std(rmse_scores)}")
        print(f"Cross-validated R²: {np.mean(r2_scores)} ± {np.std(r2_scores)}")

        # 使用完整训练集进行训练
        self.model.train(self.x_train, self.y_train)
        pred = self.model.predict(self.x_train)

        # 计算训练集上的 RMSE 和 R²
        rmse = mean_squared_error(self.y_train, pred, squared=False)
        r2 = r2_score(self.y_train, pred)
        print(f"Training RMSE: {rmse}")
        print(f"Training R²: {r2}")


    def evaluate_model(self):
        # 使用模型预测验证集的结果
        pred = self.model.predict(self.x_val)

        # 计算验证集的 RMSE 和 R²
        rmse = mean_squared_error(self.y_val, pred, squared=False)
        r2 = r2_score(self.y_val, pred)
        print(f"Validation RMSE: {rmse}")
        print(f"Validation R²: {r2}")

        # 为测试数据集准备输入和输出
        x_test = self.test_typhoons[:, :2, :].reshape(-1, 26)
        y_test = self.test_typhoons[:, 2, 5:]

        # 对测试集的输入进行缩放
        x_test = self.scale.transform(x_test)

        # 预测测试集的结果
        lr_pred = self.model.predict(x_test)

        # 计算测试集的 RMSE 和 R²
        rmse_test = mean_squared_error(y_test, lr_pred, squared=False)
        r2_test = r2_score(y_test, lr_pred)
        print(f"Test RMSE: {rmse_test}")
        print(f"Test R²: {r2_test}")


    def visualize_sample(self):
        """
        随机选取一个测试台风样本进行可视化
        返回值标注了台风路径和预测结果的地图
        """
        # 将测试集的前两部分数据取出并调整形状，以便进行标准化和预测
        x_test = self.test_typhoons[:, :2, :].reshape(-1, 26)
        y_test = self.test_typhoons[:, 2, 5:]
        # 对测试集的数据进行标准化处理
        x_test = self.scale.transform(x_test)
        # 对测试集数据进行PCA降维（注释掉）
        # x_test = self.pca.transform(x_test)
        # 随机选择一个样本的索引
        idx = random.randint(0, len(x_test) - 1)
        # 将选中的样本进行反标准化处理，并调整形状以便后续处理
        # x_sample = self.scale.inverse_transform(self.pca.inverse_transform(x_test[idx].reshape(1, -1))).reshape(2, -1)
        x_sample = self.scale.inverse_transform(x_test[idx].reshape(1, -1)).reshape(2, -1)  # 使用原始数据反标准化
        # 从测试集中取出选中样本的实际值
        y_sample = y_test[idx]
        # 使用模型对所有测试样本进行预测，并取出选中样本的预测结果
        lr_pred_sample = np.array(self.model.predict(x_test)[idx])
        # 创建一个地图对象，初始位置设置为选中样本的第一个经纬度值
        m = folium.Map(location=[x_sample[0][5], x_sample[0][6]], zoom_start=5, width=600, height=600)
        # 在地图上绘制两个起始位置、一个实际位置和一个预测位置的圆圈
        folium.Circle(location=[x_sample[0][5], x_sample[0][6]], radius=x_sample[0][11] * 1852, fill=True, color="black", fill_color="yellow").add_to(m)
        folium.Circle(location=[x_sample[1][5], x_sample[1][6]], radius=x_sample[1][11] * 1852, fill=True, color="black", fill_color="yellow").add_to(m)
        folium.Circle(location=[y_sample[0], y_sample[1]], radius=y_sample[-2] * 1852, fill=True, color="purple", fill_color="orange").add_to(m)
        folium.Circle(location=[lr_pred_sample[0], lr_pred_sample[1]], radius=lr_pred_sample[-2] * 1852, fill=True, color="blue", fill_color="red").add_to(m)
        return m


# Usage
def main():
    processor = TyphoonDataProcessor("../data/archive/typhoon_data.csv")
    processor.load_data()
    processor.split_data()
    processor.train_model()
    processor.evaluate_model()
    map_visualization = processor.visualize_sample()
    map_visualization.save("typhoon_visualization.html")
    
if __name__ == "__main__":
    main()
