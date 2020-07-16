## import modules here
import numpy as np


################# Question 1 #################


def inner_product(a, b):
    res = 0
    for i in range(len(a)):
        res += a[i] * b[i]
    return res


def get_max_index(data):
    for row in range(0, data.shape[0]):
        for column in range(0, data.shape[1]):
            if column <= row:
                data[row, column] = -1
    return np.unravel_index(data.argmax(), data.shape)


def get_element(l, result):
    for i in l:
        if isinstance(i, int):
            result.append(i)
        else:
            get_element(i, result)


def hc(data, k):  # do not change the heading of the function
    # pass # Replace this line with your implementation...
    if k == 1:
        return [0] * data.shape[0]
    if k == data.shape[0]:
        return list(range(data.shape[0]))
    length = data.shape[0]
    metrix = []
    for i in range(0, length):
        temp_l = []
        for j in range(0, length):
            temp_l.append(inner_product(data[i], data[j]))
        metrix.append(temp_l)

    # metrix = [[-1, 0.3, 0.7, 0.65, 0.2], [-1, -1, 0.7, 0.6, 0.5], [-1, -1, -1, 0.9, 0.3],
    # [-1, -1, -1, -1, 0.8], [-1, -1, -1, -1, -1]]
    metrix = np.array(metrix)
    # result = [0] * data.shape[0]

    current_cluster = range(0, data.shape[0])

    while metrix.shape[0] != 1:
        row, column = get_max_index(metrix)
        new_cluster = [current_cluster[row], current_cluster[column]]
        current_cluster = [current_cluster[index] for index in range(0, len(current_cluster)) if
                           index != row and index != column]
        current_cluster = [new_cluster] + current_cluster
        # print(metrix)
        # print("record the  vector index {} {} {}".format(row, column, current_cluster))

        if len(current_cluster) == k:
            break

        new_metrix = []
        temp_list = [-1]

        # for i in range(0, metrix.shape[0]):
        #     if i != row and i != column:
        #         previous_list = metrix[i].tolist()
        #         temp_list = [previous_list[k] for k in range(0, len(previous_list)) if k not in [row, column]]
        #
        #         value = [metrix[i][row], metrix[i][column], metrix[row][i], metrix[column][i]]
        #         while -1 in value and len(value) != 1:
        #             value.remove(-1)
        #         temp_list.append(min(value))
        #         new_metrix.append(temp_list)

        for i in range(metrix.shape[1]):
            if i != row and i != column:
                value = [metrix[i][row], metrix[i][column], metrix[row][i], metrix[column][i]]
                while -1 in value and len(value) != 1:
                    value.remove(-1)
                temp_list.append(min(value))
        new_metrix.append(temp_list)

        for i in range(metrix.shape[1]):
            if i != row and i != column:
                previous_list = metrix[i].tolist()
                temp_list = [previous_list[k] for k in range(0, len(previous_list)) if k not in [row, column]]
                new_metrix.append([-1] + temp_list)

        metrix = np.array(new_metrix)
        # # print(metrix)

    cluster = [0] * data.shape[0]
    # # print(current_cluster)

    label = 0
    for i in current_cluster:
        if isinstance(i, int):
            cluster[i] = label
            label += 1
        else:
            result = []
            get_element(i, result)
            for j in result:
                cluster[j] = label
            label += 1

    return cluster
