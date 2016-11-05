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


def discrete_data(X, y, feature_index):
    num_features = X.shape[1]
    n_classes = len(np.unique(y))
    discrete_points = []
    for i in feature_index:
        min_value = np.min(X[:, i])
        max_value = np.max(X[:, i])
        per_value = (max_value - min_value) / n_classes
        # print('+++min_value:%d, max_value:%d; n_classes:%d' %(min_value, max_value, n_classes))
        discrete_points.append([min_value + per_value * (j + 1) for j in range(n_classes - 1)])
    discrete_points = np.array(discrete_points)
    return discrete_points


def best_feature(X, y, discrete_points, n_features):
    root_entropy = info_entropy(y)
    n_samples = X.shape[0]  # sample number, feature number
    n_classes = len(np.unique(y))  # classes number
    temp_y = []
    gain = -np.inf
    best_feature_index = -1
    for i in n_features:
        index = 0
        for j in range(n_classes):
            if j == 0:
                temp_y.append(y[X[:, i] < discrete_points[index, 0]])
            elif j == n_classes - 1:
                temp_y.append(y[X[:, i] > discrete_points[index, j - 1]])
            else:
                temp_y.append(y[(X[:, i] > discrete_points[index, j - 1]) & (X[:, i] < discrete_points[index, j])])
        print('root_entropy:', root_entropy)
        temp_gain = info_gain(root_entropy, temp_y, n_samples)
        print(temp_gain)
        if temp_gain > gain:
            gain = temp_gain
            best_feature_index = i
        index += 1
    return best_feature_index


class DecisionTree:
    def __init__(self, separate_measure='info_gain', is_discrete=True):
        pass

    def fit(self, X, y, feature_names):
        n_classes = len(np.unique(y))
        if n_classes == 1:
            print('all classes are same,return!')
            return
        if len(feature_names) == 1:
            print('only one feature, return!')
            return
        discrete_points = discrete_data(X, y, feature_names)
        print('shape:', X.shape, y.shape)
        print('++++++++' , discrete_points)
        feature_index = best_feature(X, y, discrete_points, feature_names)
        combine_xy = np.c_[X, y]
        for i in range(n_classes):
            feature_i = [0, 1, 2, 3]
            index = 0
            if i == 0:
                combine_xy = combine_xy[X[:, feature_index] < discrete_points[index, i]]
            elif i == n_classes-1:
                combine_xy = combine_xy[X[:, feature_index] > discrete_points[index, i]]
            else:
                combine_xy = combine_xy[(X[:, feature_index] > discrete_points[index, i - 1]) & (
                combine_xy[:, feature_index] > discrete_points[index, i])]

            print('+++++split data feature index: %s' % feature_index)
            feature_i.pop(feature_index)
            index += 1
            self.fit(combine_xy[:, :-1], combine_xy[:, -1], feature_i)


if __name__ == '__main__':
    iris_data = load_iris()
    feature_names = list(range(len(iris_data.feature_names)))
    print(feature_names)
    X = iris_data.data
    y = iris_data.target
    dt = DecisionTree()
    dt.fit(X, y, feature_names)

