import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
import folium
import models.LinearRegression as lr
from sklearn.metrics import mean_squared_error

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
        self.model = lr.LinearRegression()
        self.scale = StandardScaler()

    def load_data(self):
        # 读取CSV数据，并调整经纬度数据的单位
        self.data = pd.read_csv(self.file_path, index_col=0)
        self.data[["Latitude of the center", "Longitude of the center"]] /= 10 # 缩成一行
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

        # 预计算台风年份，避免重复计算
        for typhoon in typhoons:
            nptyphoon = np.array(typhoon)
            typhoon_year = typhoon.iloc[0]["year"]  # 提前计算台风年份
            num_iterations = nptyphoon.shape[0] - 3  # 提前计算循环次数

            for i in range(0, num_iterations, interval):
                single_data = np.expand_dims(nptyphoon[i:i + 3], axis=0)

                # 将2022年的台风数据作为测试集
                if typhoon_year == 2022:
                    self.test_typhoons = np.append(self.test_typhoons, single_data, axis=0)
                else:
                    self.dataset = np.append(self.dataset, single_data, axis=0)

    def split_data(self):
        # 从数据集中提取特征，使用切片操作获取除了最后一列以外的所有列的前两行
        x = self.dataset[:, :2, :]
        # 从数据集中提取标签，使用切片操作获取特定列以后的所有部分
        y = self.dataset[:, 2, 5:]
        # 使用train_test_split函数将数据分为训练集和验证集，验证集占30%
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(x, y, test_size=0.3, random_state=42, shuffle=True)
        # 打印训练集和验证集的形状，以便检查数据划分情况
        print(self.x_train.shape)
        print(self.y_train.shape)
        print(self.x_val.shape)
        print(self.y_val.shape)
        # 重塑训练集和验证集的特征，以便于后续的处理或模型训练
        self.x_train = self.x_train.reshape(-1, 26)
        self.x_val = self.x_val.reshape(-1, 26)
        # 对训练集特征进行标准化处理，fit_transform同时计算和应用标准化参数
        self.x_train = self.scale.fit_transform(self.x_train)
        # 对验证集特征应用相同的标准化处理，仅使用之前计算的标准化参数进行转换
        self.x_val = self.scale.transform(self.x_val)

    def train_model(self):
        # 使用指定的参数训练模型
        self.model.train(self.x_train, self.y_train, method="matrix", learning_rate=0.1, n_iters=5000)
        pred = self.model.predict(self.x_train)
        # 计算并存储均方根误差
        rmse = mean_squared_error(pred, self.y_train, squared=False)
        print("Training Score:", rmse)

    def evaluate_model(self):
        # 使用模型预测验证集的结果
        pred = self.model.predict(self.x_val)
        # 计算验证集的均方根误差
        rmse = mean_squared_error(pred, self.y_val, squared=False)
        print("Test Score:", rmse)
        # 为测试数据集准备输入和输出
        x_test = self.test_typhoons[:, :2, :].reshape(-1, 26)
        y_test = self.test_typhoons[:, 2, 5:]
        # 对测试集的输入进行缩放
        x_test = self.scale.transform(x_test)
        # 预测测试集的结果
        lr_pred = self.model.predict(x_test)
        # 计算测试集的均方根误差并给出评分
        rmse = mean_squared_error(lr_pred, y_test, squared=False)
        print("Test Score:", rmse)

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
        # 随机选择一个样本的索引
        idx = random.randint(0, len(x_test))
        # 将选中的样本进行反标准化处理，并调整形状以便后续处理
        x_sample = self.scale.inverse_transform(x_test[idx].reshape(1, -1)).reshape(2, -1)
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