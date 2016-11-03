####  numpy一些用法

* np.unique(ar, return_index=False, return_inverse=False, return_counts=False)
```python
np.unique()去除y中重复出现的元素和每个元素重复出现的次数
unique_y, unique_y_counts = np.unique(y, return_counts=True)  # 此方法狠凶残
```

* 返回数组中满足条件的元素索引
```python
arr = np.array([1, 1, 1, 134, 45, 3, 46, 45, 65, 3, 23424, 234, 12, 12, 3, 546, 1, 2])
print(np.where(arr > 3))  # 返回arr数组中满足条件的元素索引
# print(arr > 3)
# [False False False  True  True False  True  True  True False  True  True True  True False  True False False]

y = np.zeros(arr.shape)
y[arr > 3] = 1  # 将y数组中对应满足条件的索引元素设置成1
# [ 0.  0.  0.  1.  1.  0.  1.  1.  1.  0.  1.  1.  1.  1.  0.  1.  0.  0.]
```

* np.delete(arr, obj, axis=None)
```python
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
```