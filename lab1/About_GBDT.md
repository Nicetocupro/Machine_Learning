# **梯度提升树（Gradient Boosting Decision Trees, GBDT）**

-- 它能够逐步调整每棵树的预测误差（使用多棵树进行集成学习），适合处理具有较强非线性关系的预测问题。GBDT 可以捕捉复杂的特征交互，并可能对台风路径的非线性关系有很好的建模能力。

### 集成学习（通过多个弱学习器来构建强学习器）---GBDT

1. **多个弱学习器（决策树）的训练**：
   - 在[`train`](vscode-file://vscode-app/e:/Software_environment/Vscode/Microsoft VS Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.esm.html)方法中，使用一个循环来训练多个决策树，每个树都基于前一个树的残差进行训练。
   - [`self.models`](vscode-file://vscode-app/e:/Software_environment/Vscode/Microsoft VS Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.esm.html)列表存储了所有训练好的决策树。
2. **逐步减小残差**：
   - 每个决策树都试图拟合当前的残差（即前一个树的预测误差）。
   - 通过更新残差（`residuals -= self.learning_rate * predictions`），每个新的树都在前一个树的基础上进行改进。
3. **组合多个弱学习器的预测**：
   - 在[`predict`](vscode-file://vscode-app/e:/Software_environment/Vscode/Microsoft VS Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.esm.html)方法中，通过累加所有决策树的预测结果来得到最终的预测值。
   - 每个树的预测结果都乘以学习率（[`self.learning_rate`](vscode-file://vscode-app/e:/Software_environment/Vscode/Microsoft VS Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.esm.html)），然后累加到最终的预测结果中。

[GBDT的原理、公式推导、Python实现、可视化和应用 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/280222403#:~:text=梯度提升决策树（Gradient Boosting Decision,Tree，GBDT）是一种基于boosting集成学习思想的加法模型，训练时采用前向分布算法进行贪婪的学习，每次迭代都学习一棵CART树来拟合之前 t-1 棵树的预测结果与训练样本真实值的残差。)

### 提升树procedure

先求出c1，c2切分点

```python
    # 切分数组函数
    def split_arr(self, X_data):
        self.N = X_data.shape[0]
        # 候选切分点——前后两个数的中间值
        for i in range(1, self.N):
            self.candidate_splits.append((X_data[i][0] + X_data[i - 1][0]) / 2)
        self.n_split = len(self.candidate_splits)
        # 切成两部分
        for split in self.candidate_splits:
            left_index = np.where(X_data[:, 0] <= split)[0]
            right_index = np.where(X_data[:, 0] > split)[0]
            self.split_index[split] = (left_index, right_index)
        return
```



![image-20241009212223004](C:\Users\yyd20\AppData\Roaming\Typora\typora-user-images\image-20241009212223004.png)

求各个树的方程f_1(x)、f_2(x)...

![image-20241009212419530](C:\Users\yyd20\AppData\Roaming\Typora\typora-user-images\image-20241009212419530.png)

```python
    # 计算每个切分点的误差
    def calculate_error(self, split, y_result):
        indexs = self.split_index[split]
        left = y_result[indexs[0]]
        right = y_result[indexs[1]]

        c1 = np.sum(left) / len(left)  # 左均值
        c2 = np.sum(right) / len(right)
        y_result_left = left - c1
        y_result_right = right - c2
        result = np.hstack([y_result_left, y_result_right])   # 数据拼接
        result_square = np.apply_along_axis(lambda x: x ** 2, 0, result).sum()
        return result_square, c1, c2
```

```python
    # 基于当前组合树，预测X的输出值
    def predict_x(self, X):
        s = 0
        for split, c1, c2 in zip(self.split_list, self.c1_list, self.c2_list):
            if X < split:
                s += c1
            else:
                s += c2
        return s

    # 每添加一颗回归树，就要更新y,即基于当前组合回归树的预测残差
    def update_y(self, X_data, y_data):
        y_result = []
        for X, y in zip(X_data, y_data):
            y_result.append(y - self.predict_x(X[0]))  # 残差
        y_result = np.array(y_result)
        print(np.round(y_result,2))  # 输出每次拟合训练数据的残差
        res_square = np.apply_along_axis(lambda x: x ** 2, 0, y_result).sum()
        return y_result, res_square

    def fit(self, X_data, y_data):
        self.split_arr(X_data)
        y_result = y_data
        while True:
            self.best_split(y_result)
            y_result, result_square = self.update_y(X_data, y_data)
            if result_square < self.error:
                break
        return

    def predict(self, X):
        return self.predict_x(X)
```

### GBDT算法

![image-20241009213437674](C:\Users\yyd20\AppData\Roaming\Typora\typora-user-images\image-20241009213437674.png)

![image-20241009213500665](C:\Users\yyd20\AppData\Roaming\Typora\typora-user-images\image-20241009213500665.png)

### 算法优缺点

![image-20241009213605703](C:\Users\yyd20\AppData\Roaming\Typora\typora-user-images\image-20241009213605703.png)