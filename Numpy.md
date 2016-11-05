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

#### 删除数组中符合条件的多个元素
```python
# [a[i] for i in range(len(a)) if a[i]!= 0 or i+1==len(a) or a[i+1] != 2]
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
* np.c_[X, y]  # 追加一列
* np.r_[X, x]  # 追加一行
```python
iris_data = load_iris()
feature_names = iris_data.feature_names
X = iris_data.data
y = iris_data.target
combine_xy = np.c_[X, y]  # 在X追加一列y
print(combine_xy)
```

#### 格式化输出
```python
'''
%s    字符串 (采用str()的显示)
%r    字符串 (采用repr()的显示)
%c    单个字符
%b    二进制整数
%d    十进制整数
%i    十进制整数
%o    八进制整数
%x    十六进制整数
%e    指数 (基底写为e)
%E    指数 (基底写为E)
%f    浮点数
%F    浮点数，与上相同
%g    指数(e)或浮点数 (根据显示长度)
%G    指数(E)或浮点数 (根据显示长度)
'''
```

### map/reduce
* map
```bash
>>> def f(x):
...     return x * x
...
>>> r = map(f, [1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> list(r)
[1, 4, 9, 16, 25, 36, 49, 64, 81]
```
* reduce 形如：reduce(f, [x1, x2, x3, x4]) = f(f(f(x1, x2), x3), x4)
```bash
>>> from functools import reduce
>>> def add(x, y):
...     return x + y
...
>>> reduce(add, [1, 3, 5, 7, 9])
25
```
* map/reduce
```python
from functools import reduce

# 把str转换为int的函数
def char2num(s):
    return {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}[s]

def str2int(s):
    return reduce(lambda x, y: x * 10 + y, map(char2num, s))
```
