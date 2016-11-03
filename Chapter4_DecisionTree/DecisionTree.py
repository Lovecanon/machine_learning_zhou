import numpy as np

from sklearn.datasets import load_iris

def info_entropy(y):
    len_y = len(y)
    unique_y, counts_y = np.unique(y, return_counts=True)  # 此方法狠凶残
    entropy = .0
    for c in counts_y:
        prob = c / len_y
        entropy += -prob * np.log2(prob)
    return entropy


def info_gain(pre_entropy, y, num_y):
    sum_entropy = 0
    for sub_y in y:
        sum_entropy += (len(sub_y) / num_y) * info_entropy(sub_y)
    return pre_entropy - sum_entropy


def discrete_data(X, y):
    num_features = X.shape[1]
    n_class = len(np.unique(y))
    discrete_points = []
    for i in range(0, num_features):
        min_value = np.min(X[:, i])
        max_value = np.max(X[:, i])
        per_value = (max_value - min_value) / n_class
        discrete_points.append([-np.inf, min_value + per_value, max_value - per_value, np.inf])
    discrete_points = np.array(discrete_points)
    return discrete_points


def best_feature(X, y, discrete_point):
    root_entropy = info_entropy(y)
    n_features = X.shape[1]
    n_y = len(y)
    temp_y = []
    gain = -np.inf
    best_feature_index = -1
    for i in range(n_features):
        for j in range(len(discrete_point[i]) - 1):
            temp_y.append(y[X[:, i] < discrete_point[i, j+1]])
        temp_gain = info_gain(root_entropy, temp_y, n_y)
        if temp_gain > gain:
            gain = temp_gain
            best_feature_index = i
    return best_feature_index


class DecisionTree:
    def __init__(self, separate_measure='info_gain', is_discrete=True):
        pass

    def fit(self, X, y, feature_names):
        if len(feature_names) == 1:
            print('last feature name:%s' % feature_names)

        discrete_points = discrete_data(X, y)
        best_feature_index = best_feature(X, y, discrete_points)
        print('best_feature_index %s' % best_feature_index)

        X = np.delete(X, best_feature_index, axis=1)
        feature_names = np.delete(feature_names, best_feature_index, axis=0)
        self.fit(X, y, feature_names)


if __name__ == '__main__':
    iris_data = load_iris()
    feature_names = iris_data.feature_names
    X = iris_data.data
    y = iris_data.target
    dt = DecisionTree()
    dt.fit(X, y, feature_names)
