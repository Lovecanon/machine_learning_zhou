import numpy as np

from sklearn.datasets import load_iris

def info_entropy(y):
    len_y = len(y)
    unique_y, unique_y_counts = np.unique(y, return_counts=True)  # 此方法狠凶残
    entropy = .0
    for i in range(len(unique_y)):
        prob = unique_y_counts[i] / len_y
        entropy += -prob * np.log2(prob)
    return entropy


def info_gain(pre_entropy, separate_y, num_y):
    sum_entropy = 0
    for y in separate_y:
        len_y = len(y)
        sum_entropy += (len_y / num_y) * info_entropy(y)
    return pre_entropy - sum_entropy


def discrete_data(X):
    num_features = X.shape[1]
    discrete_points = []
    for i in range(0, num_features):
        mean_value = np.mean(X[:, i])
        discrete_points.append(mean_value)
    return discrete_points


class DecisionTree:
    def __init__(self, separate_measure='info_gain', is_discrete=True):
        pass

    def fit(self, X, y, feature_names):
        num_sample, num_features = X.shape

        for j in y.shape[1]:
            # pre_pruning

            root_entropy = info_entropy(y)
            discrete_points = discrete_data(X)  # 连续值需要离散化
            separate_y = []
            max_gain = 0
            best_index = 0
            for i in range(num_features):
                separate_y.append(y[X[:, i] <= discrete_points[i]])
                separate_y.append(y[X[:, i] > discrete_points[i]])
                temp_gain = info_gain(root_entropy, separate_y, num_sample)

                if temp_gain > max_gain:
                    max_gain = temp_gain
                    best_index = i
                    print('X feature number %d, best_feature: %s' % (num_features, feature_names[best_index]))

        if X.shape[1] == 1:
            print('+++++Return')
            return

        X = np.delete(X, best_index, axis=1)
        self.fit(X, y, feature_names)


if __name__ == '__main__':
    iris_data = load_iris()
    feature_names = iris_data.feature_names
    X = iris_data.data
    y = iris_data.target
    dt = DecisionTree()
    dt.fit(X, y, feature_names)

