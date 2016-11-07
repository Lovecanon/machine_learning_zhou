# coding=utf-8
import jieba
import os
import time
from sklearn.datasets.base import Bunch
from sklearn.utils import check_random_state
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.cluster import k_means


def load_data(file_path):
    # reference@rockychi1001
    list_dirs = os.walk(file_path)
    news_name = []  # 新闻标题
    news_content_dict = {}  # 使用map防止内存溢出
    news_content = []  # 新闻内容
    news_label = []  # 新闻类别标签

    for root, dirs, files in list_dirs:
        for f in files:
            relative_path = os.path.join(root, f)
            split_path = relative_path.split(os.path.sep)
            news_name.append(split_path[-1])  # 相对路径倒数第一为文件名
            news_label.append(split_path[-2])  # 倒数第二为类别标签
            with open(relative_path, 'r', encoding='utf-8') as f:
                news_content_dict[split_path[-1]] = f.read()

    if len(news_name) != len(news_label) | len(news_label) != len(news_content):
        print('news_name and news_label and news_content are not correspond!')
        return

    #  shuffle the data
    random_state = check_random_state(0)
    indices = np.arange(len(news_name))
    random_state.shuffle(indices)
    news_name = np.array(news_name)[indices]
    news_label = np.array(news_label)[indices]

    for name in news_name:
        news_content.append(news_content_dict[name])  # 容易嘛我
    return Bunch(data=news_content, target=news_label, news_name=news_name)


def print_time(description):
    print(description, time.strftime(' : %H:%M.%S', time.localtime(time.time())))


def jieba_tokenizer(x):
    return jieba.cut(x)


def text_classifier():
    # 1.加载数据并切分数据
    news_data = load_data('./data/train')
    test_data = load_data('./data/test')
    news_content = news_data.data
    news_label = news_data.target
    X_train, X_test, y_train, y_test = train_test_split(news_content, news_label, test_size=0.001)
    print_time('loading data complete')

    # 2.将新闻内容转成词向量
    words_tfidf_vector = TfidfVectorizer(binary=False, tokenizer=jieba_tokenizer)
    words_tfidf_vector.fit(X_train)
    X_train = words_tfidf_vector.transform(X_train)
    X_test = words_tfidf_vector.transform(X_test)  # 从训练数据中分出来的测试数据
    my_test = words_tfidf_vector.transform(test_data.data)  # 新的测试数据
    print_time('convert to vector complete')

    # 3.训练分类器
    clf = LinearSVC()
    clf.fit(X_train, y_train)
    # 4.预测, 很是凶残，除了一篇讲[机器学习新书发售]预测成【教育】，其他都是正确的
    y_test_predict = clf.predict(my_test)
    print('Predict:', y_test_predict)
    print('Truth:', test_data.target)
    print_time('finish')


def text_clustering():
    # 1.加载数据并切分数据
    news_data = load_data('./data/train')
    test_data = load_data('./data/test')
    news_content = news_data.data
    news_label = news_data.target
    X_train, X_test, y_train, y_test = train_test_split(news_content, news_label, test_size=0.001)

    # 2.将新闻内容转成词向量
    words_tfidf_vector = TfidfVectorizer(binary=False, tokenizer=jieba_tokenizer)
    words_tfidf_vector.fit(X_train)
    X_train = words_tfidf_vector.transform(X_train)
    X_test = words_tfidf_vector.transform(X_test)  # 从训练数据中分出来的测试数据
    my_test = words_tfidf_vector.transform(test_data.data)  # 新的测试数据

    # k_means
    cluster = k_means(X_train, n_clusters=11)


if __name__ == '__main__':
    text_classifier()

