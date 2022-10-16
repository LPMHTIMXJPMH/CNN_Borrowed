a = 1

def foo(a):
    a += 1

foo(a)
print(a)

import numpy as np

b = np.array([[1,2],[3,4]])

c = np.array([[2,1],[4,3]])

c = np.dot(b, c)
print(c)
c = np.dot(c, b)
print(c)

d = np.matmul(b, c)
print(d)
d = np.matmul(c, b)
print(d)

import torch
e = torch.tensor(
    [[1,2],
    [3,4],
    [5,6]]
)

print(torch.sum(e, dim = 1 ))


numpy_image = np.array([1,1,1,1,1,1])
print(numpy_image.T)

torch_image = torch.tensor([1,1,1,1,1,1])
print(torch_image.T)

weights = np.ones((6, 128))
print(weights.T.shape)
print(numpy_image.shape)
print(np.dot(weights.T, numpy_image))


f = np.array([1,2,3,4,5,6])
f = f.reshape((-1,1))
g = np.array([1,2,3,4])
g = g.reshape(1,-1)
h = np.matmul(f, g)
print(h.shape)