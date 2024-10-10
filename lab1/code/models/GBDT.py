import numpy as np

class SimpleDecisionTreeRegressor:
    def __init__(self, maxdepth=3):
        self.maxdepth = maxdepth
        self.tree = None

    def fit(self, X, y):
        self.tree = self.buildtree(X, y, depth=0)

    def predict(self, X):
        return np.array([self.predictsingle(x, self.tree) for x in X])

    def buildtree(self, X, y, depth):
        if depth >= self.maxdepth or len(set(y)) == 1:
            return np.mean(y)
        
        bestsplit = self.findbestsplit(X, y)
        if bestsplit is None:
            return np.mean(y)
        
        leftindices = X[:, bestsplit['feature']] < bestsplit['threshold']
        rightindices = ~leftindices
        
        lefttree = self.buildtree(X[leftindices], y[leftindices], depth + 1)
        righttree = self.buildtree(X[rightindices], y[rightindices], depth + 1)
        
        return {'feature': bestsplit['feature'], 'threshold': bestsplit['threshold'], 'left': lefttree, 'right': righttree}

    def findbestsplit(self, X, y):#找到最佳分割点
        bestsplit = None
        bestmse = float('inf')
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                leftindices = X[:, feature] < threshold
                rightindices = ~leftindices
                
                if len(y[leftindices]) == 0 or len(y[rightindices]) == 0:
                    continue
                
                mse = self.calculatemse(y[leftindices], y[rightindices])
                if mse < bestmse:
                    bestmse = mse
                    bestsplit = {'feature': feature, 'threshold': threshold}
        
        return bestsplit

    def calculatemse(self, lefty, righty):#计算均方误差
        leftmse = np.var(lefty) * len(lefty)
        rightmse = np.var(righty) * len(righty)
        return (leftmse + rightmse) / (len(lefty) + len(righty))

    def predictsingle(self, x, tree):
        if not isinstance(tree, dict):
            return tree
        
        if x[tree['feature']] < tree['threshold']:
            return self.predictsingle(x, tree['left'])
        else:
            return self.predictsingle(x, tree['right'])

class GBDT:
    def __init__(self, nestimators=100, learningrate=0.1, maxdepth=3):
        self.nestimators = nestimators
        self.learningrate = learningrate
        self.maxdepth = maxdepth
        self.models = []

    def train(self, X, y):
        residuals = y.copy()
        
        for _ in range(self.nestimators):
            tree = SimpleDecisionTreeRegressor(maxdepth=self.maxdepth)
            tree.fit(X, residuals)
            predictions = tree.predict(X)
            residuals -= self.learningrate * predictions
            self.models.append(tree)

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        
        for tree in self.models:
            predictions += self.learningrate * tree.predict(X)
        
        return predictions