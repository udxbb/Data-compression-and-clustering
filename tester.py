import numpy as np
import submission as submission
from scipy.cluster.hierarchy import linkage, fcluster


def dot_product(a, b):
    res = 0
    for i in range(len(a)):
        res += a[i] * b[i]
    return res


def inverse_dot_product(a, b):
    res = 0
    for i in range(len(a)):
        res += a[i] * b[i]
    return 1 / res


data = np.loadtxt('asset/data.txt', dtype = float)
# data = np.array([[1, 0.9, 0.1, 0.65, 0.2], [0.9, 1, 0.7, 0.6, 0.5], [0.1, 0.7, 1, 0.4, 0.3],
# [0.65, 0.6, 0.4, 1, 0.8], [0.2, 0.5, 0.3, 0.8, 1]])
k = 1
result = submission.hc(data, k)

Z = linkage(data, 'complete', metric=inverse_dot_product)
right = fcluster(Z, k, 'maxclust')

print(result)
print(right)




def compute_error(data, labels, k):
    n, d = data.shape
    centers = []
    for i in range(k):
        centers.append([0 for j in range(d)])

    for i in range(n):
        centers[labels[i]] = [x + y for x, y in zip(centers[labels[i]], data[i])]

    error = 0
    for i in range(n):
        error += dot_product(centers[labels[i]], data[i])
    return error


error = compute_error(data, submission.hc(data, k), k)

print(error)
# print(cluster)
