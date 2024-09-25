import numpy as np
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
from models.LinearRegression import LinearRegression
from models.RidgeRegression import RidgeRegression
from models.LassoRegression import LassoRegression  # 确保你有这个模型的实现


def test_simple_linear_regression():
    w = 1.34
    b = 2.08
    size = 100
    X = np.expand_dims(np.linspace(-10, 10, size), axis=1)
    y = X.flatten() * w + b + (np.random.random(size) - 1) * 3

    lr1 = LinearRegression()
    lr1.train(X, y, method='formula')
    print("【formula方法】\nw:{}, b:{}, square loss:{}".format(lr1.w, lr1.b, lr1.sqrLoss))

    lr2 = LinearRegression()
    lr2.train(X, y, method='Matrix')
    print("【matrix方法】\nw:{}, b:{}, square loss:{}".format(lr2.w, lr2.b, lr2.sqrLoss))

    fig, ax = plt.subplots()
    ax.scatter(X, y)
    ax.plot(X, X * lr2.w + lr2.b, color='r', linewidth=3)
    plt.show()


def test_multivariate_regressions():
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    # 测试线性回归的矩阵法
    lr1 = LinearRegression()
    lr1.train(X, y, method='Matrix')
    print("【matrix方法】\nw:{}, b:{}, square loss:{}".format(lr1.w, lr1.b, lr1.sqrLoss))

    # 测试线性回归的梯度下降法
    lr2 = LinearRegression()
    lr2.train(X, y, method='Gradient', learning_rate=0.1, n_iters=5000)
    print("【gradient方法】\nw:{}, b:{}, square loss:{}".format(lr2.w, lr2.b, lr2.sqrLoss))

    # 画梯度下降的误差下降图
    fig, ax = plt.subplots()
    ax.plot(range(lr2.n_iters), lr2.lossList, linewidth=3)
    ax.set_title("Square Loss for Gradient Descent")
    plt.xlabel("Iterations")
    plt.ylabel("Square Loss")
    plt.show()

    # 测试岭回归
    ridge = RidgeRegression()
    ridge.train(X, y, lambdas=0.2)
    print("【Ridge回归】\nw:{}, b:{}, square loss:{}".format(ridge.w, ridge.b, ridge.sqrLoss))

    # 测试Lasso回归（坐标下降法）
    lasso_cd = LassoRegression()
    lasso_cd.train(X, y, method='coordinate_descent', lambdas=0.001, n_iters=1000)
    print("【Lasso回归 - 坐标下降法】\nw:{}, b:{}, square loss:{}".format(lasso_cd.w, lasso_cd.b, lasso_cd.sqrLoss))

    # 绘制坐标下降法的误差下降图
    plt.figure()
    plt.plot(range(len(lasso_cd.lossList)), lasso_cd.lossList, label='Coordinate Descent Loss')
    plt.title("Lasso Regression - Coordinate Descent Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Square Loss")
    plt.legend()
    plt.show()

    # 测试Lasso回归（最小角回归法）
    lasso_lars = LassoRegression()
    lasso_lars.train(X, y, method='least_angle_regression', lambdas=0.001, n_iters=1000)
    print("【Lasso回归 - 最小角回归法】\nw:{}, b:{}, square loss:{}".format(lasso_lars.w, lasso_lars.b, lasso_lars.sqrLoss))

    # 绘制最小角回归法的误差下降图
    plt.figure()
    plt.plot(range(len(lasso_lars.lossList)), lasso_lars.lossList, label='Least Angle Regression Loss')
    plt.title("Lasso Regression - Least Angle Regression Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Square Loss")
    plt.legend()
    plt.show()


def run_tests():
    test_simple_linear_regression()
    test_multivariate_regressions()


if __name__ == "__main__":
    run_tests()
