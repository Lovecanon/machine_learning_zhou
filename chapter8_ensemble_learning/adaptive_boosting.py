from numpy import *
import matplotlib.pyplot as plt


def load_data_arr(file_name):
    """
        加载西瓜3.0alpha 数据，密度、含糖量、好瓜坏瓜。
        hsplit(data_arr, (a, b))按列切割(竖直切割)
            0<= column <a、a<= column <b、column >=b,把数据分成三部分；
            hsplit(data_arr, (a,))，把数据分成两部分。
    :param file_name: 文件名
    :return:
    """
    data_set = []
    with open(file_name, 'r') as fr:
        for line in fr.readlines():
            data_set.append(list(map(float, line.strip('\n').split(' '))))  # map/reduce
    data_arr = array(data_set)
    label_arr = hsplit(data_arr, (2,))[-1][:, 0]  # [:, 0] 将返回的二维数组转成一维数组
    data_arr = hsplit(data_arr, (2,))[0]  # 返回一个二维数组
    return data_arr, label_arr


if __name__ == '__main__':
    data_arr, label_arr = load_data_arr('../data/data_set3.0alpha.txt')
    print(data_arr)
    print(label_arr)
