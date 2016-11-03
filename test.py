import numpy as np
from sklearn.datasets import load_iris

arr = np.arange(1, 7).reshape(2, 3)
print(arr)
# [[1 2 3]
# [4 5 6]]
a = np.delete(arr, [1, 3], axis=1)  # 删除索引为1和3的列， 这里没有索引为3的列故只删除第二列
print(a)
# [[1 3]
# [4 6]]

b = np.delete(arr, 1, axis=0)  # 删除索引值为1的行
print(b)
# [[1 2 3]]