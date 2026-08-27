from collections import defaultdict

import hypernetx as hnx
import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import itertools
import pandas as pd
from pathlib import Path
from WS_load_hypergraph_from_json import load_hypergraph_from_json

# Change this file when running a different artificial hypergraph dataset.
HYPERGRAPH_FILE = Path(__file__).resolve().parent / "BA_uniform_hypergraph.json"
H = load_hypergraph_from_json(HYPERGRAPH_FILE)

ALL_HYPEREDGES_PROBABILITIES = (0.3, 0.6, 0.9)
DEFAULT_ALL_HYPEREDGES_PROBABILITY = 0.5

# 我现在需要补充超参数实验。也就是在文件sir_model、sir_model_email_w3c、Artificial_high_order_network_sir中有一个参数，在传播时判断节点是感染他的邻居还是随机选择一条超边传播。
# # 需要删除
# def find_closest_node(H, observer_nodes, top_n=30):
#     # 从超图 H 创建普通图 G
#     G = hypergraph_to_graph(H)
#
#     # 计算每个节点到观察节点的最短路径
#     shortest_paths = {}
#     for node in G.nodes():
#         if node not in observer_nodes:
#             shortest_paths[node] = min(
#                 (nx.shortest_path_length(G, source=node, target=obs_node) for obs_node in observer_nodes),
#                 default=np.inf
#             )
#
#     # 选择离观察节点集合最近的前 top_n 个节点
#     closest_nodes = sorted(shortest_paths.items(), key=lambda x: x[1])[:top_n]
#
#     # 返回节点及其距离
#     return closest_nodes

# 将超图转换为包含所有边对的图

def hypergraph_to_graph(H):
    edge_list = []
    for edge_id, edge in enumerate(H.edges):
        # 添加所有的二元组（即 2-单纯形）
        edge_list.extend(itertools.combinations(sorted(list(H.edges[edge])), 2))
    # # 打印单纯形网络的边
    # print("单纯形网络的边:")
    # for edge in edge_list:
    #     print(edge)

    # 将边列表转换为图
    G = nx.Graph()
    G.add_edges_from(edge_list)

    return G

def create_node_to_edge_dict(H):
    node_to_edge_dict = {node: set() for node in H.nodes}
    for edge_id, edge in enumerate(H.edges):
        for node in sorted(list(H.edges[edge])):
            node_to_edge_dict[node].add(edge_id)
    return node_to_edge_dict

# 计算边的特征向量的分数
def count_adj_hypeedges(num_hyperedges):

    # 获取超边个数，形成方阵
    num_rows_columns = num_hyperedges
    # 创建全零矩阵
    zero_matrix = np.zeros((num_rows_columns, num_rows_columns))

    edge_sets = [set(H.edges[edge]) for edge in H.edges]

    for edge_out_id, out_edge in enumerate(H.edges):

        for edge_in_id, in_edge in enumerate(H.edges):
            # print(edge_out_id)
            # print(edge_in_id)
            # print(sorted(list(H.edges[out_edge])))
            # 主对角线元素为0
            if edge_out_id == edge_in_id:
                zero_matrix[edge_out_id, edge_in_id] = 0
            else:
                # 将两个列表转换为集合，求交集
                common_elements = edge_sets[edge_out_id] & edge_sets[edge_in_id]
                # 计算相同元素的个数
                zero_matrix[edge_out_id, edge_in_id] = len(common_elements)

    # 求解特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(zero_matrix)

    # 找到最大特征值对应的索引
    max_eigenvalue_index = np.argmax(eigenvalues)

    # 取出最大特征值对应的特征向量
    max_eigenvector = eigenvectors[:, max_eigenvalue_index]

    # 计算数组的绝对值
    abs_max_eigenvector = np.abs(max_eigenvector)

    return abs_max_eigenvector

def count_average_nodes(vector):
    # 定义字典，存储节点的重要性
    nodes_dict = defaultdict(float)
    for node, edges in node_to_edge_dict.items():
            # print(node)
            # print(edges)
            for element in edges:
                # print(element)
                # print(vector[element])
                nodes_dict[node] += vector[element]
    # print(nodes_dict)
    # print(1)
    # 选择前三个键
    top_keys = sorted(nodes_dict, key=nodes_dict.get, reverse=True)

    # 限制为前三个键
    top_keys = top_keys[:3]
    return top_keys

def simulate_SIR(
    H, node_to_edge_dict, beta, gamma, initial_infected,
    all_hyperedges_probability=DEFAULT_ALL_HYPEREDGES_PROBABILITY,
):
    if not 0 <= all_hyperedges_probability <= 1:
        raise ValueError("all_hyperedges_probability must be between 0 and 1")

    if isinstance(initial_infected, int):
        initial_infected = [initial_infected]

    # 初始化状态
    S, I, R = 'S', 'I', 'R'
    states = {node: S for node in H.nodes}
    infection_record = {node: {'infected_time': None, 'infected_by': None} for node in H.nodes}  # 感染记录
    for i in initial_infected:
        states[i] = I
        infection_record[i]['infected_time'] = 0

    susceptible_counts = []
    infected_counts = []
    recovered_counts = []
    infected_nodes = set(initial_infected)
    # print(infected_nodes)
    recovered_nodes = set()
    t = 1
    while any(state == I for state in states.values()):
        new_states = states.copy()

        for node in H.nodes:
            if states[node] == I:
                # 感染邻居节点
                if node in node_to_edge_dict:
                    # print(node_to_edge_dict)
                    # 全局和局部的传播方式（公开和私密）
                    if random.random() < all_hyperedges_probability:
                        for edge in node_to_edge_dict[node]:
                            for neighbor in H.edges[edge]:
                                if states[neighbor] == S and random.random() < beta:
                                    new_states[neighbor] = I
                                    infection_record[neighbor]['infected_time'] = t
                                    infection_record[neighbor]['infected_by'] = node
                                    infected_nodes.add(neighbor)
                                    t += 1
                    else:
                        random_edge = random.choice(list(node_to_edge_dict[node]))
                        for neighbor in H.edges[random_edge]:
                            if states[neighbor] == S and random.random() < beta:
                                new_states[neighbor] = I
                                infection_record[neighbor]['infected_time'] = t
                                infection_record[neighbor]['infected_by'] = node
                                infected_nodes.add(neighbor)
                                t += 1

                # 自身恢复
                if random.random() < gamma:
                    new_states[node] = R
                    recovered_nodes.add(node)
                    infected_nodes.discard(node)

        states = new_states

        # 统计每个状态的节点数量
        susceptible_counts.append(sum(1 for state in states.values() if state == S))
        infected_counts.append(sum(1 for state in states.values() if state == I))
        recovered_counts.append(sum(1 for state in states.values() if state == R))

    return susceptible_counts, infected_counts, recovered_counts, infection_record, infected_nodes, recovered_nodes

def save_csv(nums, infection_record, obs, path):

    new_list = nums

    time = list()
    prior_node = list()
    for node in obs:
        infected_time = infection_record[node].get('infected_time')
        infected_by = infection_record[node].get('infected_by')

        # 检查 infected_time 是否有效
        if infected_time is not None:
            time.append(infected_time)
        else:
            time.append(None)  # 或者可以选择跳过这个值

        # 检查 infected_by 是否有效
        if infected_by is not None:
            prior_node.append(infected_by)
        else:
            prior_node.append(None)  # 或者可以选择跳过这个值
    if len(new_list) > len(obs):
        max_length = len(new_list)
        extended_obs = obs + [0] * (max_length - len(obs))
        extended_time = time + [0] * (max_length - len(time))
        extended_prior_node = prior_node + [0] * (max_length - len(prior_node))
    else:
        max_length = len(obs)
        new_list = new_list + [0] * (max_length - len(new_list))
        extended_obs = obs
        extended_time = time
        extended_prior_node = prior_node

    dateframe = pd.DataFrame({
        "id_i": new_list,
        "id_obs": extended_obs,
        "obs_time": extended_time,
        "obs_pd": extended_prior_node
    })

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    dateframe.to_csv(path, index=False)

# 将超图转换成单纯形网络计算（其他算法适用）
G = hypergraph_to_graph(H)
# print(G.nodes)
# print(G.edges)
# print(G)

# 创建节点到超边的映射
node_to_edge_dict = create_node_to_edge_dict(H)

# Observer nodes depend only on the loaded hypergraph, so calculate them once
# and reuse them for every source, probability, and repeated snapshot.
num_hyperedges = len(H.edges)
hypeedge_important = count_adj_hypeedges(num_hyperedges)
OBSERVER_NODES = count_average_nodes(hypeedge_important)

# # 打印节点到超边的映射
# print("\n节点到超边的映射:")
# for node, edges in node_to_edge_dict.items():
#     print(f"节点 {node} 属于超边: {edges}")

def main(
    path, item,
    all_hyperedges_probability=DEFAULT_ALL_HYPEREDGES_PROBABILITY,
):
    # SIR 模型参数
    beta = 0.3  # 感染概率
    gamma = 0.2  # 恢复率

    # # 计算并打印超边数量

    # 获取边的特征向量值
    # print(hypeedge_important)
    #  分配给节点，再进行特征向量值计算
    obsever_nodes = OBSERVER_NODES
    # print(obsever_nodes)
    # # 获取离观察节点集合最近的前30个节点及其距离
    # closest_nodes = find_closest_node(H, obsever_nodes, top_n=30)
    # print("离观察节点集合最近的前30个节点及其距离:")
    # for node, distance in closest_nodes:
    #     print(f"节点 {node}: 距离 {distance}")

    # initial_infected = random.choice(list(H.nodes))
    initial_infected = item

    # print("源:", initial_infected)

    # 模拟 SIR 传播
    susceptible_counts, infected_counts, recovered_counts, infection_record, infected_nodes, recovered_nodes = simulate_SIR(
        H, node_to_edge_dict, beta, gamma, initial_infected,
        all_hyperedges_probability)
    # print(susceptible_counts, infected_counts, recovered_counts)

    # print(infected_nodes, recovered_nodes)
    # 将集合转换为列表，并按相同顺序合并相加
    merged_get_information = list(infected_nodes) + list(recovered_nodes)

    # print(merged_get_information)

    # for node in obsever_nodes:
    #     print(infection_record[node]['infected_time'])
    #     print(infection_record[node]['infected_by'])

    save_csv(merged_get_information, infection_record, obsever_nodes, path)

    # # 绘制 SIR 传播过程
    # plt.figure(figsize=(10, 6))
    # plt.plot(susceptible_counts, label='Susceptible (S)', color='blue')
    # plt.plot(infected_counts, label='Infected (I)', color='red')
    # plt.plot(recovered_counts, label='Recovered (R)', color='green')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Number of Nodes')
    # plt.title('SIR Model on a Small-World Hypergraph')
    # plt.legend()
    # plt.show()

if __name__ == "__main__":

    # # BA
    # rand_source = [(325, 470), (322, 10)]
    # # BA_N
    # rand_source = [(3, 2), (2, 1)]

    # # WS_N
    # rand_source = [(28, 86), (86, 120), (120, 167), (167, 188), (188, 223), (223, 304), (304, 347), (347, 413), (413, 488), (488, 28)]

    # WS
    # rand_source = [(28, 32), (32, 47), (47, 110), (116, 135), (135, 263), (263, 352), (352, 366), (366, 367), (367, 482), (482, 28)]
    # The same 10 single-source nodes used by the previous 0.5/0.5
    # single-source experiment.
    rand_source = [0, 1, 4, 34, 39, 94, 104, 119, 335, 452]

    for item in rand_source:
        print(item)
        for all_hyperedges_probability in ALL_HYPEREDGES_PROBABILITIES:
            one_hyperedge_probability = 1 - all_hyperedges_probability
            output_dir = (
                f'BA_uniform_snapshot_all{all_hyperedges_probability:.1f}_'
                f'one{one_hyperedge_probability:.1f}'
            )
            for i in range(10):
                path = (
                    f'{output_dir}/BA-{item}-{i}-new.csv'
                )
                main(path, item, all_hyperedges_probability)
