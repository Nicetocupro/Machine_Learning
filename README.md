# Project: Machine Learning Lab Experiments

## Overview
This repository contains three lab experiments focused on machine learning techniques, implemented and documented with code and reports.

### Labs Included:
1. **Lab 1: Regression Experiment**  
   - Implementation of regression models and analysis of their performance.

2. **Lab 2: Classification Experiment**  
   - Classification techniques applied to different datasets with evaluation metrics.

3. **Lab 3: Clustering Experiment**  
   - Exploration of clustering algorithms to uncover hidden patterns in data.
i
## Structure
-i `Lab1/`: Code and report for the regression experiment.
- `Lab2/`: Code and report for the classification experiment.
- `Lab3/`: Code and report for the clustering experiment.

## How to Run
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-repo-url
   ```
2. Navigate to the specific lab directory:
   ```bash
   cd Lab1  # or Lab2 / Lab3
   For lab1,there is diffrent models
   Usage
   To run the script with different models, use the following commands:
      python main.py --model linear
      python main.py --model ridge
      python main.py --model lasso
   ```
3. Run ithe corresponding code with your preferred Python environment.

## How to push to Github

以下是这些步骤的详细说明：

### 1. 在 GitHub 上创建新分支
首先，你需要在 GitHub 上基于上一个分支创建一个新的分支。例如，如果 `lab2` 是基于 `main` 分支的，你可以创建 `lab2-init` 基于 `lab2`。这一步是为了确保新分支的基础代码是最新的、稳定的。

### 2. 在本地拉取最新代码
在本地终端中运行以下命令，确保你的仓库内容是最新的：

```bash
git pull
```

### 3. 在本地切换到新分支
使用以下命令，在本地创建并切换到一个新分支。此分支名称可以是任何你想要的新分支名，例如 `new_branch`。

```bash
git checkout -b new_branch origin/new_branch
```

### 4. 添加更改的文件
在开发过程中，将更新的文件添加到暂存区，以便进行提交。例如，如果你更新了 `lab2/pages` 下的文件，可以使用以下命令：

```bash
git add lab2/pages
```

### 5. 提交更改
为这些更改添加注释并提交。`""` 中是你的提交信息，例如 "添加新的页面内容"。

```bash
git commit -m "添加新的页面内容"
```

### 6. 推送更改
将本地分支推送到远程仓库：

```bash
git push
```

### 7. 将新分支合并到上一个分支
当你完成所有开发并且新分支已经准备好，可以将新分支合并回上一个分支（例如 `main` 或 `lab2`）。



