import numpy as np
import sklearn.datasets, sklearn.linear_model
import matplotlib.pyplot as plt


class Config:
    nn_input_dim = 2  # 输入层数据维数
    nn_output_dim = 2  # 输出层数据维数
    # Gradient descent parameters
    epsilon = 0.01  # learning rate for gradient descent
    reg_lambda = 0.01  # regularization strength


def generate_data():
    # Generate a dataset and plot it
    # The dataset we generated has two classes, plotted as red and blue points.
    np.random.seed(0)
    X, y = sklearn.datasets.make_moons(200, noise=0.20)
    return X, y


def plot_decision_boundary(pred_func, X, y):
    """
        绘制图像
    :param pred_func: 边界函数
    :param X: 数据集
    :param y: 类别标签
    :return:
    """
    # 设置最小最大值, 加上一点外边界
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # 根据最小最大值和一个网格距离生成整个网格
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # 对整个网格预测边界值
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # 绘制边界和数据集的点
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()


def visualize(X, y, model):
    plot_decision_boundary(lambda x: predict(model, x), X, y)
    plt.title('Logistic Regression')


def calculate_loss(model, X, y):
    """
        计算整个模型的性能
    :param model: 训练模型
    :param X: 数据集
    :param y: 类别标签
    :return: 误判的概率
    """
    num_examples = len(X)  # 样本总数
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # 正向传播来计算预测的分类值
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # 计算误判概率
    correct_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(correct_logprobs)
    # 加入正则项修正错误(可选)
    data_loss += Config.reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1. / num_examples * data_loss


def predict(model, x):
    """
        预测类别属于(0 or 1)
    :param model: 训练模型
    :param x: 预测向量
    :return: 判决类别
    """
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # 正向传播计算
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)


def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False):
    """
        生成一个指定层数的神经网络模型
    :param X:  200*2 数据集
    :param y:  类别标签
    :param nn_hdim: 隐藏层层数
    :param num_passes: 迭代次数
    :param print_loss: 是否输出误判率
    :return: 神经网络模型
    """
    num_examples = len(X)
    # 根据维度随机初始化参数
    np.random.seed(0)
    W1 = np.random.randn(Config.nn_input_dim, nn_hdim) / np.sqrt(Config.nn_input_dim)  # 2*3 矩阵
    b1 = np.zeros((1, nn_hdim))  # 1*3 矩阵
    W2 = np.random.randn(nn_hdim, Config.nn_output_dim) / np.sqrt(nn_hdim)  # 3*2 矩阵
    b2 = np.zeros((1, Config.nn_output_dim))  # 1*2 矩阵

    # This is what we return at the end
    model = {}

    # Gradient descent. For each batch...
    for i in range(0, num_passes):

        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)  # 原始归一化
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # 后向传播
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # 加入修正项 (b1 and b2 don't have regularization terms)
        dW2 += Config.reg_lambda * W2
        dW1 += Config.reg_lambda * W1

        # 更新梯度下降参数
        W1 += -Config.epsilon * dW1
        b1 += -Config.epsilon * db1
        W2 += -Config.epsilon * dW2
        b2 += -Config.epsilon * db2

        # 更新模型
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        # 一定迭代次数后输出当前误判率
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 1000 == 0:
            print("Loss after iteration %i: %f" % (i, calculate_loss(model, X, y)))
    return model


def classify(X, y):
    # 使用Logistic Regression生成的边界函数
    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(X, y)
    return clf


def main():
    X, y = generate_data()
    # Build a model with a 3-dimensional hidden layer
    model = build_model(X, y, 3, print_loss=True)
    visualize(X, y, model)

if __name__ == "__main__":
    # 代码参考@dennybritz
    main()