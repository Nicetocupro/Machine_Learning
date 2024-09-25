import numpy as np
from sklearn.datasets import fetch_california_housing  # 添加这一行
import matplotlib.pyplot as plt
from models.LinearRegression import LinearRegression


def simpleLR(w, b, size=100):
    X = np.expand_dims(np.linspace(-10, 10, size), axis=1)
    y = X.flatten() * w + b + (np.random.random(size) - 1) * 3
    # 公式法求解
    lr1 = LinearRegression()
    lr1.train(X, y, method='formula')
    print("【formula方法】\nw:{}, b:{}, square loss:{}".format(lr1.w, lr1.b, lr1.sqrLoss))
    # 矩阵法求解
    lr2 = LinearRegression()
    lr2.train(X, y, method='Matrix')
    print("【matrix方法】\nw:{}, b:{}, square loss:{}".format(lr2.w, lr2.b, lr2.sqrLoss))
    # 画图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(X, y)
    ax.plot(X, X * lr2.w + lr2.b, color='r', linewidth=3)
    plt.show()
    return


def multivariateLR():
    # X, y = load_boston(True)  # 移除这一行
    housing = fetch_california_housing()  # 加载加州房价数据集
    X, y = housing.data, housing.target  # 提取特征和目标变量

    # 将特征X标准化，方便收敛
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    # 矩阵法求解
    lr1 = LinearRegression()
    lr1.train(X, y, method='Matrix')
    print("【matrix方法】\nw:{}, b:{}, square loss:{}".format(lr1.w, lr1.b, lr1.sqrLoss))

    # 梯度下降法求解
    lr2 = LinearRegression()
    lr2.train(X, y, method='Gradient', learning_rate=0.1, n_iters=5000)
    print("【gradient方法】\nw:{}, b:{}, square loss:{}".format(lr2.w, lr2.b, lr2.sqrLoss))

    # 画梯度下降的误差下降图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(lr2.n_iters), lr2.lossList, linewidth=3)
    ax.set_title("Square Loss")
    plt.show()
    return


if __name__ == "__main__":
    # 1、先用公式法和矩阵法测试下一元线性回归
    simpleLR(1.34, 2.08)
    # 2、再用矩阵法和梯度下降法测试下多元线性回归
    multivariateLR()
