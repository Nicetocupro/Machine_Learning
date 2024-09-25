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
        self.data = pd.read_csv(self.file_path, index_col=0)
        self.data["Latitude of the center"] /= 10
        self.data["Longitude of the center"] /= 10
        self.data = self.data.drop(["Indicator of landfall or passage"], axis=1)
        self.data = self.data[self.data["year"] >= 1977]
        idx = self.data["grade"] == "Just entering into the responsible area of RSMC Tokyo-Typhoon Center"
        self.data = self.data.drop(self.data[idx].index)
        self.data = self.data.replace({
            "Tropical Depression": "TD",
            "Severe Tropical Storm": "STS",
            "Tropical Storm": "TS",
            "Extra-tropical Cyclone": "L",
            "Typhoon": "TY"
        })
        self.data = self.data.dropna()
        self.data = self.data.drop(["Direction of the longest radius of 50kt winds or greater", "Direction of the longest radius of 30kt winds or greater"], axis=1)
        grade_map = {"TD": 1, "TS": 2, "STS": 3, "TY": 4}
        self.data["grade"] = self.data["grade"].map(grade_map)
        self.data = self.data[self.data["hour"] % 6 == 0]

        ids = self.data["International number ID"].unique()
        typhoons = [self.data[self.data["International number ID"] == ID].drop(["International number ID"], axis=1) for ID in ids]

        interval = 1
        self.dataset = np.empty((0, 3, 13))
        self.test_typhoons = np.empty((0, 3, 13))

        for typhoon in typhoons:
            nptyphoon = np.array(typhoon)
            for i in range(0, nptyphoon.shape[0] - 3, interval):
                single_data = nptyphoon[i:i + 3]
                single_data = np.expand_dims(single_data, axis=0)
                if typhoon.iloc[0]["year"] == 2022:
                    self.test_typhoons = np.append(self.test_typhoons, single_data, axis=0)
                    continue
                self.dataset = np.append(self.dataset, single_data, axis=0)

    def split_data(self):
        x = self.dataset[:, :2, :]
        y = self.dataset[:, 2, 5:]
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(x, y, test_size=0.3, random_state=42, shuffle=True)
        print(self.x_train.shape)
        print(self.y_train.shape)
        print(self.x_val.shape)
        print(self.y_val.shape)
        self.x_train = self.x_train.reshape(-1, 26)
        self.x_val = self.x_val.reshape(-1, 26)
        self.x_train = self.scale.fit_transform(self.x_train)
        self.x_val = self.scale.transform(self.x_val)

    def train_model(self):
        self.model.train(self.x_train, self.y_train, method="matrix", learning_rate=0.1, n_iters=5000)
        pred = self.model.predict(self.x_train)
        rmse = mean_squared_error(pred, self.y_train, squared=False) # 计算均方根误差
        print("Training Score:", rmse)

    def evaluate_model(self):
        pred = self.model.predict(self.x_val)
        rmse = mean_squared_error(pred, self.y_val, squared=False)
        print("Test Score:", rmse)

        x_test = self.test_typhoons[:, :2, :].reshape(-1, 26)
        y_test = self.test_typhoons[:, 2, 5:]
        x_test = self.scale.transform(x_test)
        lr_pred = self.model.predict(x_test)
        rmse = mean_squared_error(lr_pred, y_test, squared=False)
        print("Test Score:", rmse)

    def visualize_sample(self):
        x_test = self.test_typhoons[:, :2, :].reshape(-1, 26)
        y_test = self.test_typhoons[:, 2, 5:]
        x_test = self.scale.transform(x_test)
        idx = random.randint(0, len(x_test))
        x_sample = self.scale.inverse_transform(x_test[idx].reshape(1, -1)).reshape(2, -1)
        y_sample = y_test[idx]
        lr_pred_sample = np.array(self.model.predict(x_test)[idx])

        m = folium.Map(location=[x_sample[0][5], x_sample[0][6]], zoom_start=5, width=600, height=600)
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