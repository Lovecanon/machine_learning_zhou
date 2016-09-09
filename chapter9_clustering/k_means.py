from numpy import *
import matplotlib.pylab as plt


def load_data_arr(file_name):
    """
        加载西瓜4.0数据，含有密度、含糖量两个feature。
    :param file_name: 文件名
    :return:
    """
    data_set = []
    with open(file_name, 'r') as fr:
        for line in fr.readlines():
            data_set.append(list(map(float, line.strip('\n').split(' '))))  # map/reduce
    return array(data_set)


def get_centroids(data_arr, k):
    '''
        随机产生k个质心
    :param data_arr: 数据集
    :param k: k个质心
    :return:
    '''
    data_size = len(data_arr)
    centroids_index = list(map(int, random.uniform(0, high=data_size, size=k)))  # int直接去掉小数部分，并非四舍五入
    centroids = data_arr[centroids_index, :]
    return centroids


def distance_euclidean(vector_a, vector_b):
    return sqrt(sum(power(vector_a - vector_b, 2)))  # 欧几里德距离


def k_means(data_arr, k=3):
    '''
        k-means算法
        step1：随机产生3个质心点，并初始化数据-簇矩阵(cluster_and_distance)；
        step2：在每个数据点上计算到三个质心距离，取最小距离保存到cluster_and_distance；
        step3：如果最小距离有变化，重新计算三个质心点坐标，直到最小距离不再变化。
    :param data_arr: 数据集
    :param k: 这里k用3代替
    :return:
    '''
    centroids = get_centroids(data_arr, k)
    has_change = True  # 如果最小距离有变动则继续迭代
    r, c = shape(data_arr)
    cluster_and_distance = zeros((r, 2))  # 存放数据属于哪个质心信息。【质心索引，距离】
    cluster_and_distance[:, -1] = inf
    while has_change:
        has_change = False
        for i in range(len(data_arr)):
            for j in range(len(centroids)):  # 将数据分入离自己最近的质点中
                distance_i_j = distance_euclidean(data_arr[i], centroids[j])
                if distance_i_j < cluster_and_distance[i, -1]:
                    cluster_and_distance[i] = [j, distance_i_j]
                    has_change = True
        print(centroids)
        if has_change:  # 重新计算质心位置
            for centroid in range(len(centroids)):
                current_centroid_data = data_arr[nonzero(cluster_and_distance[:, 0] == centroid)[0]]
                print(len(current_centroid_data))
                centroids[centroid, :] = mean(current_centroid_data, axis=0)
        print('++++++')
    return centroids, cluster_and_distance


def show_data(data_arr, centroids, cluster_and_distance):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    data_arr = column_stack((data_arr, cluster_and_distance[:, 0]))  # 扩展一列，表示该条数据属于某一簇
    ax.scatter(centroids[:, 0], centroids[:, 1], s=40, marker='+')
    # 这一句有点屌，根据最后一列簇类别来给每个数据点上色
    ax.scatter(data_arr[:, 0], data_arr[:, 1], c=100.0 * array(data_arr[:, -1]), marker='s')
    plt.show()


if __name__ == '__main__':
    data_arr = load_data_arr('../data/data_set4.0.txt')
    centroids, cluster_and_distance = k_means(data_arr)
    show_data(data_arr, centroids, cluster_and_distance)