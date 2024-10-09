# 实验报告
## 一、课题综述

### 1.1 课题说明

本课题的研究对象是台风数据的回归分析，旨在通过机器学习中的回归方法对台风的关键指标（如风速、气压、路径等）进行建模和预测，从而提高对台风影响的理解和预测能力。

台风作为一种严重的自然灾害，其强度和路径的准确预测对于防灾减灾、社会经济发展和公共安全都有重要意义。然而，由于台风运动的复杂性和多变性，传统的气象预测方法在面对大量数据和复杂特征时难以提供高精度的预测结果。因此，本课题利用机器学习中的回归技术，尝试通过以下几种回归模型来实现台风的关键参数预测：

1. **线性回归（Linear Regression）**：用于探索台风数据中各特征变量与目标变量（如风速、气压等）之间的线性关系。

2. **Lasso 回归（Lasso Regression）**：通过增加 L1 正则化项，可以对特征进行稀疏选择，有效减少过拟合并提高模型的可解释性。

3. **Ridge 回归（Ridge Regression）**：通过增加 L2 正则化项，减少模型对噪声和多重共线性问题的敏感性，提高模型的泛化能力。

综上所述，本课题旨在利用不同的回归模型分析和预测台风的关键特征，提供一种新的技术手段，为更精确的台风预测提供支持。

### 1.2 课题目标

本课题的具体目标分为以下几个方面：

**模型构建与实现**：
   - **手写实现**线性回归、Lasso 回归和 Ridge 回归模型，通过代码编写深入理解各个模型的损失函数、梯度下降和正则化等核心机制。

**模型性能分析与优化**：
   - 探讨**学习率**对模型收敛速度和稳定性的影响，选择最优的学习率参数。
   - 研究**正则化参数**（L1 和 L2）对模型权重分布、过拟合控制和模型复杂度的影响，通过调整正则化参数优化模型性能。

**特征工程与数据分析**：
   - **分析输入变量之间的关系**，通过相关矩阵、热力图等方法研究不同变量之间的相关性，筛选出对模型预测有较大影响的特征。
   - 进行**数据降维和升维**的尝试，探索主成分分析（PCA）等降维方法在数据处理中的应用，以及通过特征交互项、特征组合等方法对数据进行升维处理，提升模型的表现。

**模型训练与评估**：
   - 利用划分好的训练集和测试集，分别对不同回归模型进行训练和测试，并使用均方误差（MSE）、决定系数（R²）等指标对模型性能进行评估。

**结果可视化与解释**：
   - 可视化模型预测结果与实际数据的对比，展示模型在不同条件下的表现。
   - 解释模型中各个特征对预测结果的贡献，帮助理解模型预测的内在机制。

**总结与展望**：
   - 总结不同回归模型的优缺点，分析其在台风数据预测中的表现。
   - 为未来更复杂模型（如深度学习、集成学习）的研究提供参考。

通过本课题的研究，我们期望能够在理解和预测台风的多种特征变量上取得进展，为台风强度和路径的准确预测提供理论支持和实践指导。

### 1.3 课题数据集

本课题使用的数据集来源于 Kaggle 网站上的台风数据集，该数据集包括台风的历史数据和相关气象特征，用于探索和预测台风的强度和路径。数据集的详细信息如下：

1. **数据集简介**：
   - 数据集包含台风在不同时间点的详细观测记录，包括风速、气压、经纬度等多种气象特征。这些特征数据可用于训练机器学习模型，预测台风的强度和未来路径。
   - 数据文件包括：
     - `typhoon_data.csv`：每个台风在不同时刻的详细记录，包括时间、风速、气压、地理位置等。
     - `typhoon_info.csv`：台风的总体信息，包括台风编号、名称、时间范围、最高风速、最低气压等。

2. **数据集特征**：
   - `typhoon_data.csv` 文件的主要特征描述：
     - **Year**：年份，表示台风数据记录的年份。
     - **Month**：月份，表示台风数据记录的月份。
     - **Day**：日期，表示台风数据记录的日期。
     - **Hour**：小时，表示台风数据记录的小时。
     - **Lat**：台风中心的纬度位置（单位：度）。
     - **Lon**：台风中心的经度位置（单位：度）。
     - **Pressure**：台风中心的气压（单位：hPa）。
     - **Wind Speed**：台风中心的最大风速（单位：m/s）。
     - **Typhoon Category**：台风的分类级别（如热带风暴、台风、强台风等）。
     - **Typhoon ID**：台风编号，用于唯一标识每个台风事件。

   - `typhoon_info.csv` 文件的主要特征描述：
     - **Typhoon ID**：台风编号，与 `typhoon_data.csv` 中的 `Typhoon ID` 对应，用于唯一标识每个台风事件。
     - **Name**：台风名称。
     - **Start Date**：台风开始时间，表示该台风事件的初始观测时间。
     - **End Date**：台风结束时间，表示该台风事件的最后观测时间。
     - **Max Wind Speed**：台风在其生命周期中达到的最大风速（单位：m/s）。
     - **Min Pressure**：台风在其生命周期中达到的最低气压（单位：hPa）。
     - **Duration**：台风持续时间，以小时为单位。

3. **数据预处理**：
   - **数据清洗**：包括处理缺失值、重复值和异常值。比如，对于少量的缺失气压或风速值，可以采用前后时刻的平均值进行填充；对于明显异常的地理位置数据需要进行删除或修正。
   - **数据合并**：将 `typhoon_data.csv` 和 `typhoon_info.csv` 进行合并，形成包含每个台风完整生命周期和相关特征的综合数据集，以便后续模型训练和分析。
   - **时间序列处理**：将时间特征（年、月、日、小时）组合成标准的时间戳格式，以便进行时间序列预测。

4. **数据集划分**：
   - 将完整的数据集按照时间或者随机方式划分为训练集和测试集。通常使用 70%-80% 的数据作为训练集，其余部分作为测试集。这样既能保证模型的训练效果，又能有效地评估模型的泛化性能。

5. **特征工程**：
   - **特征提取**：从原始特征中提取更多有意义的特征，例如：
     - 计算风速和气压的变化率（增量）。
     - 根据台风的经纬度计算其移动速度和方向。
   - **特征选择**：使用相关性分析、共线性分析等方法挑选对目标变量预测影响较大的特征，去除冗余或无关特征。
   - **数据升维和降维**：对于复杂特征，可以使用特征组合或交互项来进行升维处理；对于冗余特征，可以使用主成分分析（PCA）进行降维处理，提高模型训练效率。

6. **数据集的优势与挑战**：
   - **优势**：
     - 数据集涵盖了台风的多种气象特征和时间序列信息，具有较高的时间和空间分辨率。
     - 该数据集经过详细整理和标注，为模型训练和特征工程提供了良好的基础。
   - **挑战**：
     - 台风数据具有很强的时间和空间相关性，这对模型的时序特征提取和建模能力提出了较高要求。
     - 数据不平衡问题较为严重，某些类别的台风数量明显少于其他类别，可能导致模型对某些类别的预测偏差。

## 二、实验报告设计

### 2.1 数据准备

本阶段的目的是加载和清理台风数据，为后续的模型训练做准备。

1. **数据加载**：
   - 使用 `pandas` 库从 `typhoon_data.csv` 文件中加载数据。
   - 初始数据表中包含台风的多种气象特征（如风速、气压、经纬度等）和时间戳。

2. **数据清理**：
   - 通过检查数据的基本信息，处理缺失值和重复值。采用前后时刻的平均值填补 `Wind Speed` 和 `Pressure` 中的少量缺失数据。
   - 使用 `dropna()` 方法删除严重缺失的行或列。

3. **数据合并与处理**：
   - 将 `typhoon_data.csv` 与 `typhoon_info.csv` 合并，形成包含完整台风信息的数据集。
   - 将时间特征（年、月、日、小时）组合为标准的时间戳格式，方便进行时间序列分析。

4. **数据集划分**：
   - 使用 `train_test_split` 将数据集划分为训练集和测试集，其中训练集占比 80%，测试集占比 20%。

### 2.2 数据预处理

本阶段的目标是对数据进行标准化和特征工程处理。

1. **数据标准化**：
   - 使用 `StandardScaler` 对 `Wind Speed`、`Pressure` 等特征进行标准化处理，以消除不同特征之间的量纲差异，提高模型训练的效率和稳定性。

2. **特征工程**：
   - 通过特征组合创建新的变量，如根据经纬度计算台风的移动速度和方向。
   - 针对 `Typhoon Category` 等类别型特征，使用独热编码（One-Hot Encoding）将类别转换为数值型数据。

3. **特征降维**：
   - 使用主成分分析（PCA）对高维特征数据进行降维处理，提取主要特征成分，减少模型的计算复杂度。

4. **时间序列处理**：
   - 为时间序列数据创建滞后特征和滚动平均特征，以捕捉时间维度上的变化趋势，提高模型的预测精度。

### 2.3 模型搭建

本阶段的目标是手动实现回归模型，包括线性回归和岭回归。

1. **线性回归模型**：
   - 实现了一个 `LinearRegression` 类，用于简单线性回归和多元线性回归的建模。模型的损失函数为均方误差（MSE），采用梯度下降法进行优化。
   - `fit` 方法用于模型训练，通过多次迭代更新模型参数（斜率 `w` 和截距 `b`），最小化损失函数。

2. **岭回归模型**：
   - 实现了一个 `RidgeRegression` 类，通过在损失函数中加入 L2 正则化项（`lambdas`），来限制模型参数的大小，从而减少过拟合的风险。
   - 使用 `train` 方法对模型进行训练，通过矩阵运算和最小二乘法更新模型参数。

3. **模型训练框架**：
   - 在 `main.py` 中定义了统一的模型训练框架，包括数据加载、模型初始化、训练和评估的流程。该框架可以扩展到其他回归模型。

### 2.4 模型训练测试

本阶段的目标是使用训练数据训练模型，并在测试数据上进行性能评估。

1. **模型训练**：
   - 调用 `LinearRegression` 和 `RidgeRegression` 模型的 `fit` 和 `train` 方法，使用训练数据进行模型训练。
   - 在训练过程中，记录每次迭代的损失值变化，以评估模型的收敛情况。

2. **模型调参**：
   - 针对不同模型，调整学习率、迭代次数和正则化系数等超参数，选择最优组合。
   - 使用交叉验证（Cross-Validation）方法评估不同超参数组合的性能，确定最优参数设置。

3. **模型测试与评估**：
   - 在测试集上使用训练好的模型进行预测，计算预测值与实际值的均方误差（MSE）和决定系数（R²）。
   - 通过残差分析，评估模型的预测误差分布和模型拟合效果。

4. **模型对比与选择**：
   - 比较线性回归和岭回归模型的性能，分析不同模型在不同数据特征下的表现，选择最优模型。

### 2.5 结果可视化

本阶段通过可视化手段展示模型的预测结果和特征重要性。

1. **预测结果可视化**：
   - 绘制实际值与预测值的对比图，如散点图、折线图，直观展示模型的预测效果。

2. **残差分析图**：
   - 绘制残差图和误差直方图，分析模型在不同特征值下的预测误差情况。

3. **特征重要性图**：
   - 使用特征系数绘制条形图，展示特征在模型预测中的重要性，帮助理解模型的预测逻辑。

### 2.6 分析与优化

本阶段的目标是分析模型的优缺点，并提出改进策略。

1. **模型表现分析**：
   - 分析不同回归模型的优缺点，线性回归模型简单直观，但在面对多重共线性时可能会有较差的表现；岭回归模型通过引入正则化项，有效减小了多重共线性和过拟合的影响。

2. **优化策略**：
   - 进一步优化特征工程，如引入更多与台风路径相关的动态特征（如风速变化率）。
   - 调整模型的超参数，如学习率、迭代次数和正则化系数，提升模型的泛化能力。

3. **未来研究方向**：
   - 探索更复杂的机器学习模型（如集成学习和深度学习）在台风预测中的应用，提高模型的预测精度和稳定性。


### 3. 总结

本实验通过对台风数据的回归分析，构建并手动实现了线性回归和岭回归模型。我们在数据预处理阶段对数据进行了清洗和标准化处理，并通过特征工程提升了模型的输入质量。实验中，我们详细分析了学习率和正则化参数对模型性能的影响，最终在测试数据上获得了较好的预测效果，尤其是岭回归模型在处理多重共线性问题时表现出更高的精度和稳定性。可视化分析展示了模型的预测结果和特征重要性，进一步揭示了台风特征与预测结果之间的关系。总体而言，本实验验证了回归模型在台风预测中的有效性，为未来更复杂模型的研究和优化提供了基础。