import math

import numpy as np
from collections import defaultdict
from collections import Counter
from heapq import nlargest

def read_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [list(map(int, line.strip().split(','))) for line in lines]

# 计算超边个数
def count_nodes_hypeedges():
    # data = read_file("data/hyperedges-contact-primary-school.txt")
    # data = read_file("data/hyperedges-house-committees.txt")
    data = read_file("data/hyperedges-senate-committees.txt")
    # data = read_file("data/hyperedges-senate-bills.txt")

    # 去除重复的子列表
    unique_data = list(set(tuple(sublist) for sublist in data))
    # 合并所有子列表成一个单一的列表
    flat_list = [item for sublist in unique_data for item in sublist]

    # 使用Counter计算每个元素的数量
    element_counts = Counter(flat_list)

    # 按数量降序排列
    sorted_counts = sorted(element_counts.items(), key=lambda x: x[1], reverse=True)

    print(sorted_counts)

# 计算边的特征向量的分数
def count_adj_hypeedges():
    data = read_file("data/hyperedges-contact-primary-school.txt")
    # data = read_file("data/hyperedges-house-committees.txt")
    # data = read_file("data/hyperedges-senate-committees.txt")
    # data = read_file("data/hyperedges-senate-bills.txt")

    # 去除重复的子列表
    unique_data = list(set(tuple(sublist) for sublist in data))
    # print(unique_data)
    # print(1)
    # 获取超边个数，形成方阵
    num_rows_columns = len(unique_data)
    # 创建全零矩阵
    zero_matrix = np.zeros((num_rows_columns, num_rows_columns))

    # # 获取行数和列数
    # num_rows, num_columns = zero_matrix.shape
    # # # 输出结果
    # print("行数:", num_rows)
    # print("列数:", num_columns)

    for index_out, value_out in enumerate(unique_data):
        # print(f"Index: {index_out}, Value: {value_out}")
        for index_int, value_int in enumerate(unique_data):
            # print(value_out)
            # print(value_int)
            # 使用set求交集，即两个列表的相同元素
            common_elements_count = len(set(value_out) & set(value_int))
            # print(common_elements_count)
            zero_matrix[index_out, index_int] = common_elements_count
            # print(zero_matrix[index_out, index_int])
    # 求解特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(zero_matrix)

    # 找到最大特征值对应的索引
    max_eigenvalue_index = np.argmax(eigenvalues)

    # 取出最大特征值对应的特征向量
    max_eigenvalue = eigenvalues[max_eigenvalue_index]
    max_eigenvector = eigenvectors[:, max_eigenvalue_index]

    # 计算数组的绝对值
    abs_max_eigenvector = np.abs(max_eigenvector)

    # 找到特征向量中的最大值和其所在的行号
    max_element_in_eigenvector = np.max(max_eigenvector)
    row_number_of_max_element = np.argmax(max_eigenvector)

    # 输出结果
    # print("最大特征值:", max_eigenvalue)
    # print("\n最大特征值对应的特征向量:")
    # print(max_eigenvector)
    # print(len(max_eigenvector))

    # print("特征向量中的最大值:", max_element_in_eigenvector)
    # print("最大值所在的行号:", row_number_of_max_element)

    return abs_max_eigenvector

# 分配给节点（特征向量）
def count_average_nodes(vector):
    # data = read_file("data/hyperedges-contact-primary-school.txt")
    # data = read_file("data/hyperedges-house-committees.txt")
    data = read_file("data/hyperedges-senate-committees.txt")
    # data = read_file("data/hyperedges-senate-bills.txt")

    # 去除重复的子列表
    unique_data = list(set(tuple(sublist) for sublist in data))
    # 获取节点个数和列数
    num_nodes = max(max(row) for row in unique_data)

    # 定义字典，存储节点的重要性
    nodes_dict = defaultdict(float)
    for node in range(1, num_nodes + 1):
        for columns, sublist in enumerate(unique_data):
            count_sublist = len(sublist)
            if node in sublist:
                nodes_dict[node] += vector[columns]

    # print(dict(nodes_dict))
    # 获取所有键值对按值的绝对值大小排序的结果
    sorted_nodes = sorted(nodes_dict.items(), key=lambda x: abs(x[1]), reverse=True)

    # 将节点数除以3并向上取整
    condition = math.ceil(num_nodes / 3)
    sub_condition = math.ceil(condition / 2)
    # print(condition)
    # print(sub_condition)
    count = 0
    observers = set()
    for max_node in sorted_nodes:
        max_value = nodes_dict[max_node]
        print("排序:", count, {max_node: max_value})
        count += 1
        if count in [sub_condition, sub_condition + condition, sub_condition + condition * 2]:
            # print("排序:", count, {max_node: max_value})
            observers.add(max_node[0])
    print(observers)
    # top_three_nodes = nlargest(3, nodes_dict, key=nodes_dict.get)
    # # 遍历前三个最大值，获取键和对应的值
    # for max_node in top_three_nodes:
    #     max_value = nodes_dict[max_node]
    #     print("最大值:", {max_node: max_value})


def main():
    # # 计算超边个数，并降序筛选
    # count_nodes_hypeedges()
    # 获取边的特征向量值
    hypeedge_important = count_adj_hypeedges()
    # print(hypeedge_important[0])
    # 分配给节点，再进行特征向量值计算
    count_average_nodes(hypeedge_important)


if __name__ == '__main__':
    main()

# from sir_model import networks_sir
# def main():
#     # 获取高阶网络数据集，并将其转化为一般的网络进行流动传播
#     G = networks_sir().handle_data()
#     # 给定的三个节点
#     # nodes_to_check = [101, 10, 157]
#     nodes_to_check = [131, 99]
#
#     # 用于存储邻居节点的集合
#     neighbors_sets = [set(G.neighbors(node)) for node in nodes_to_check]
#
#     # 找到重复的邻居节点
#     common_neighbors = set.intersection(*neighbors_sets)
#
#     print(f"Nodes {nodes_to_check} have common neighbors: {common_neighbors}")
# #
# if __name__ == '__main__':
#     main()