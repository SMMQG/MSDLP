import math
import json
import random
import networkx as nx
import os
import re
import pandas as pd
import numpy as np
from scipy import sparse
# from sir_model import networks_sir
from matplotlib import pyplot as plt
from itertools import combinations
# from Artificial_high_order_network_sir import H, G
from collections import deque


# =============================================================================
# 数据集配置（中文说明）
# =============================================================================
# 当前按照你的要求运行 emailW3C 数据集。
USE_EMAIL_W3C = True
EMAIL_SNAPSHOT_RATIO = 1  # 可改为 0.25、0.50、0.75 或 1.00

# 下面这些配置保留原来的 sc 数据集设置。
# 如果以后要恢复旧实验，把 USE_EMAIL_W3C 改为 False 即可。
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if USE_EMAIL_W3C:
    DATA_FILE_PATH = os.path.join(SCRIPT_DIR, "data", "email-W3C.txt")
    SNAPSHOT_DIR = os.path.join(
        SCRIPT_DIR,
        f"emailW3C_snapshot_{int(EMAIL_SNAPSHOT_RATIO * 100)}"
    )
    OUTPUT_PATH = os.path.join(
        SCRIPT_DIR,
        f"emailW3C_locating_result_{int(EMAIL_SNAPSHOT_RATIO * 100)}",
        "MSDLP.csv"
    )
else:
    # 旧版默认配置：sc 数据集。这里不删除，方便以后切回原实验。
    DATA_FILE_PATH = os.path.join(
        SCRIPT_DIR, "data", "hyperedges-senate-committees.txt"
    )
    SNAPSHOT_DIR = os.path.join(SCRIPT_DIR, "sc_snapshot")
    OUTPUT_PATH = os.path.join(
        SCRIPT_DIR, "sc_locating_result_50", "MSDLP.csv"
    )

# emailW3C 的快照是单源传播，因此默认只输出排名第一的节点。
# 如果以后运行原来的双源实验，把它改成 2 即可。
SOURCE_COUNT = 1
VERBOSE = False

# New snapshot configuration.  Set these three values before each run.
DATASET = "emailW3C"
ALL_HYPEREDGES_PROBABILITY = 0.3
SNAPSHOT_RATIO = 25

DATASET_CONFIG = {
    "emailW3C": ("email-W3C.txt", "emailW3C", "emailW3C", "text"),
    "cps": ("hyperedges-contact-primary-school.txt", "cps", "cps", "text"),
    "hc": ("hyperedges-house-committees.txt", "hc", "hc", "text"),
    "sc": ("hyperedges-senate-committees.txt", "sc", "sc", "text"),
    "sb": ("hyperedges-senate-bills.txt", "sb", "sb", "text"),
    "WS_uniform": ("WS_uniform_hypergraph.json", "WS_uniform", "WS", "json"),
    "WS_Non-uniform": ("WS_Non-uniform_hypergraph.json", "WS_Non-uniform", "WS", "json"),
    "BA_uniform": ("BA_uniform_hypergraph.json", "BA_uniform", "BA", "json"),
    "BA_Non-uniform": ("BA_Non-uniform_hypergraph.json", "BA_Non-uniform", "BA", "json"),
}

if DATASET not in DATASET_CONFIG:
    raise ValueError(f"Unsupported DATASET: {DATASET}")
if not 0 < ALL_HYPEREDGES_PROBABILITY < 1:
    raise ValueError("ALL_HYPEREDGES_PROBABILITY must be between 0 and 1")
if SNAPSHOT_RATIO not in (25, 50, 75, 100):
    raise ValueError("SNAPSHOT_RATIO must be one of 25, 50, 75, 100")

DATA_FILE_NAME, SNAPSHOT_FOLDER_PREFIX, SNAPSHOT_PREFIX, DATASET_FORMAT = DATASET_CONFIG[DATASET]
DATA_FILE_PATH = (
    os.path.join(SCRIPT_DIR, DATA_FILE_NAME)
    if DATASET_FORMAT == "json"
    else os.path.join(SCRIPT_DIR, "data", DATA_FILE_NAME)
)
PROBABILITY_LABEL = (
    f"all{ALL_HYPEREDGES_PROBABILITY:.1f}_"
    f"one{1 - ALL_HYPEREDGES_PROBABILITY:.1f}"
)
SNAPSHOT_DIR = os.path.join(
    SCRIPT_DIR,
    f"{SNAPSHOT_FOLDER_PREFIX}_snapshot_{PROBABILITY_LABEL}_{SNAPSHOT_RATIO}",
)
FULL_SNAPSHOT_DIR = os.path.join(
    SCRIPT_DIR,
    f"{SNAPSHOT_FOLDER_PREFIX}_snapshot_{PROBABILITY_LABEL}",
)
APPLY_SNAPSHOT_RATIO_IN_ALGORITHM = False
if not os.path.isdir(SNAPSHOT_DIR) and os.path.isdir(FULL_SNAPSHOT_DIR):
    SNAPSHOT_DIR = FULL_SNAPSHOT_DIR
    APPLY_SNAPSHOT_RATIO_IN_ALGORITHM = True

OUTPUT_PATH = os.path.join(
    SCRIPT_DIR,
    f"{SNAPSHOT_FOLDER_PREFIX}_locating_result_{PROBABILITY_LABEL}_{SNAPSHOT_RATIO}",
    "MSDLP.csv",
)
USE_EMAIL_W3C = DATASET == "emailW3C"

# Set to True to process every available dataset/parameter/ratio snapshot
# directory in one run.  Missing directories are skipped.
RUN_ALL_AVAILABLE_EXPERIMENTS = False
ALL_EXPERIMENT_PROBABILITIES = (0.3, 0.6, 0.9)
ALL_EXPERIMENT_SNAPSHOT_RATIOS = (25, 50, 75, 100)


def configure_experiment(dataset, all_hyperedges_probability, snapshot_ratio):
    """Update global paths and graph-loading settings for one experiment."""
    global DATASET, ALL_HYPEREDGES_PROBABILITY, SNAPSHOT_RATIO
    global DATA_FILE_NAME, SNAPSHOT_FOLDER_PREFIX, SNAPSHOT_PREFIX
    global DATASET_FORMAT, DATA_FILE_PATH, PROBABILITY_LABEL
    global SNAPSHOT_DIR, FULL_SNAPSHOT_DIR, APPLY_SNAPSHOT_RATIO_IN_ALGORITHM
    global OUTPUT_PATH, USE_EMAIL_W3C

    if dataset not in DATASET_CONFIG:
        raise ValueError(f"Unsupported DATASET: {dataset}")
    if not 0 < all_hyperedges_probability < 1:
        raise ValueError("all_hyperedges_probability must be between 0 and 1")
    if snapshot_ratio not in (25, 50, 75, 100):
        raise ValueError("snapshot_ratio must be one of 25, 50, 75, 100")

    DATASET = dataset
    ALL_HYPEREDGES_PROBABILITY = all_hyperedges_probability
    SNAPSHOT_RATIO = snapshot_ratio
    (
        DATA_FILE_NAME,
        SNAPSHOT_FOLDER_PREFIX,
        SNAPSHOT_PREFIX,
        DATASET_FORMAT,
    ) = DATASET_CONFIG[DATASET]
    DATA_FILE_PATH = (
        os.path.join(SCRIPT_DIR, DATA_FILE_NAME)
        if DATASET_FORMAT == "json"
        else os.path.join(SCRIPT_DIR, "data", DATA_FILE_NAME)
    )
    PROBABILITY_LABEL = (
        f"all{ALL_HYPEREDGES_PROBABILITY:.1f}_"
        f"one{1 - ALL_HYPEREDGES_PROBABILITY:.1f}"
    )
    SNAPSHOT_DIR = os.path.join(
        SCRIPT_DIR,
        f"{SNAPSHOT_FOLDER_PREFIX}_snapshot_"
        f"{PROBABILITY_LABEL}_{SNAPSHOT_RATIO}",
    )
    FULL_SNAPSHOT_DIR = os.path.join(
        SCRIPT_DIR,
        f"{SNAPSHOT_FOLDER_PREFIX}_snapshot_{PROBABILITY_LABEL}",
    )
    APPLY_SNAPSHOT_RATIO_IN_ALGORITHM = False
    if not os.path.isdir(SNAPSHOT_DIR) and os.path.isdir(FULL_SNAPSHOT_DIR):
        SNAPSHOT_DIR = FULL_SNAPSHOT_DIR
        APPLY_SNAPSHOT_RATIO_IN_ALGORITHM = True

    OUTPUT_PATH = os.path.join(
        SCRIPT_DIR,
        f"{SNAPSHOT_FOLDER_PREFIX}_locating_result_"
        f"{PROBABILITY_LABEL}_{SNAPSHOT_RATIO}",
        "MSDLP.csv",
    )
    USE_EMAIL_W3C = DATASET == "emailW3C"

# # # 获取高阶网络数据集，并将其转化为一般的网络进行流动传播
# G = networks_sir().handle_data()
# # 创建一个空的无向图
# G = nx.Graph()
#
# # 假设我们有一个边的列表，每个元素是一个元组，格式为 (节点1, 节点2, 权重)
# edges = [
#     (1, 2, 1), (1, 3, 1), (2, 3, 1),
#     (2, 5, 2), (2, 6, 2), (5, 6, 2),
#     (1, 7, 3), (1, 8, 3), (1, 9, 3), (7, 8, 3), (7, 9, 3), (8, 9, 3),
#     (3, 10, 4), (3, 11, 4), (3, 12, 4), (10, 11, 4), (10, 12, 4), (11, 12, 4),
#     (9, 20, 5), (9, 21, 5), (20, 21, 5),
#     (6, 17, 6), (6, 18, 6), (17, 18, 6),
#     (4, 5, 7), (4, 13, 7), (4, 14, 7), (5, 13, 7), (5, 14, 7), (13, 14, 7),
#     (14, 15, 8), (14, 16, 8), (15, 16, 8),
#     (18, 19, 9)
# ]
#
# # 使用列表中的边和权重批量添加到图中
# for edge in edges:
#     G.add_edge(edge[0], edge[1], weight=edge[2])
# 可视化网络
# def visualize_graph(G):
#     pos = nx.spring_layout(G)
#     nx.draw(G, pos, with_labels=True, font_weight='bold')
#     plt.show()
class networks_sir:
    def read_file(self, file_path):
        # 同时兼容旧数据集的逗号分隔，以及 emailW3C 的空格/制表符分隔。
        # 例如："1,2,3"、"1 2 3" 和 "1\t2\t3" 都可以读取。
        connections = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                values = re.split(r'[\s,]+', line)
                connections.append(list(map(int, values)))
        return connections

    def create_graph(self, connections):
        G = nx.Graph()
        add_weight = 1
        for connection in connections:
            for i in range(len(connection)):
                for j in range(i + 1, len(connection)):
                    G.add_edge(connection[i], connection[j], weight=add_weight)
                    # print(G.edges)
            add_weight += 1
        return G

    def visualize_graph(self, G):
        pos = nx.spring_layout(G)  # You can use different layout algorithms
        nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_size=8)
        plt.show()
    def handle_data(self):
        # 当前使用上方配置中的 emailW3C 数据文件。
        # 原来的旧数据文件路径已经保留在配置区的 else 分支中。
        if DATASET_FORMAT == "json":
            with open(DATA_FILE_PATH, "r", encoding="utf-8") as file:
                hypergraph_data = json.load(file)
            connections = list(hypergraph_data.get("edges", {}).values())
        else:
            connections = self.read_file(DATA_FILE_PATH)
        graph = self.create_graph(connections)
        # self.visualize_graph(graph)
        return graph

def get_MSDLP_source(lg, obs, pd):
    alpha = 0.5  # alpha值
    beta = 0.95  # beta值
    # 原来这里会打印全部节点和边；emailW3C 数据量较大，暂时关闭。
    # print(lg.nodes())
    # print(lg.edges())
    # print(obs, pd)
    if not obs and not pd:
        # 创建一个字典存储节点的权重值
        weights = {}
        # 输出节点编号
        # print("节点编号:")
        for node in lg.nodes():
            weights[node] = 1
    else:
        # 初始化结果集合为obs字典的所有值
        case1 = set(obs.values())
        # # 遍历第一个字典的值
        # for value in obs.values():
        #     # 获取与当前值相邻的节点
        #     neighbors = [n for n in lg.neighbors(value)]
        #     # 检查相邻节点是否不在第二个字典的值中
        #     for neighbor in neighbors:
        #         if neighbor not in pd.values():
        #             case1.add(neighbor)
        case2 = set(pd.values())
        # print("与第一个字典中的值相邻且不包含在第二个字典中的节点编号:", case1)
        # print("case2", case2)
        # num_nodes = lg.number_of_nodes()

        # 创建一个字典存储节点的权重值
        weights = {}
        # 输出节点编号
        # print("节点编号:")
        for node in lg.nodes():
            # 如果节点属于case1集合，则赋予权重值1/3
            if node in case1:
                weights[node] = 1/2
            # # 如果节点属于case2集合，则赋予权重值2/3
            # elif node in case2:
            #     weights[node] = 2/3
            # 否则，默认赋予权重值为1
            else:
                weights[node] = 1
            # print(node)
        # print("节点权重:", weights)
    # print(lg.nodes())
    # 创建矩阵
    Y_matrix = np.array([weights[node] for node in weights.keys()]).reshape(-1, 1)
    # print(Y_matrix)
    # visualize_graph(lg)
    # 获取邻接矩阵
    # 使用与节点排序一致的稀疏邻接矩阵，避免 emailW3C 生成数 GB 的稠密矩阵。
    sorted_nodes = list(lg.nodes())
    adj_matrix = nx.adjacency_matrix(lg, nodelist=sorted_nodes).tocsr()
    # print(adj_matrix)
    # 将邻接矩阵转换为二进制形式
    # 原算法只使用连接是否存在，因此将稀疏矩阵中的非零值统一为 1。
    binary_adj_matrix = adj_matrix.astype(float)
    binary_adj_matrix.data[:] = 1.0
    # print("二进制权重矩阵:")
    # print(binary_adj_matrix)

    # 计算对角矩阵D
    # D = np.diag(np.sum(binary_adj_matrix, axis=1))

    # 计算D的逆平方根矩阵D^{-1/2}
    # 稀疏计算 D^{-1/2}，同时避免孤立节点导致除零。
    degree = np.asarray(binary_adj_matrix.sum(axis=1)).ravel()
    degree_sqrt_inv = np.zeros_like(degree, dtype=float)
    nonzero_degree = degree > 0
    degree_sqrt_inv[nonzero_degree] = 1 / np.sqrt(degree[nonzero_degree])
    D_sqrt_inv = sparse.diags(degree_sqrt_inv, format='csr')

    # 计算S = D^{-1/2}WD^{-1/2}
    S = D_sqrt_inv @ binary_adj_matrix @ D_sqrt_inv
    # print(S)

    # lst = []
    # for node in lg.nodes():
    #     lst.append(node)
    # # 查找字典值在列表中的索引
    # index_map = {key: lst.index(value) for key, value in obs.items()}
    # index_2map = {key: lst.index(value) for key, value in pd.items()}
    # # print(index_map)
    # # print(index_2map)
    # # 遍历两个字典的键和值
    # for key in index_map.keys():
    #     value1 = index_map[key]
    #     value2 = index_2map[key]
    #     # 观察点编号所在行、对应感染节点所在列
    #     S[value1][value2] *= (beta / alpha)

    # 初始化节点值
    GZ = Y_matrix.copy()
    alpha = 0.3  # alpha值
    tolerance = 1e-6  # 容差
    max_iterations = 1000  # 最大迭代次数
    # for node in lg.nodes():
    #     for neighbor in lg.adj[node]:
    #         # 检查值是否存在
    #         if my_dict.get(key) == value:
    # 迭代更新节点值
    # print(2)
    for _ in range(max_iterations):
        G_new = alpha * S.dot(GZ) + (1 - alpha) * Y_matrix
        if np.linalg.norm(G_new - GZ) < tolerance:
            break
        GZ = G_new

    # print("收敛后的节点值:", GZ)

    # guess_s_index = np.argmax(GZ)
    # # 获取节点编号排序索引为1的节点
    # sorted_nodes = list(lg.nodes())
    # guess_s = sorted_nodes[guess_s_index]
    # # print(1)

    flattened_GZ = GZ.flatten()
    sorted_indices = np.argsort(flattened_GZ)[::-1]
    # new_guess = sorted_nodes[sorted_indices[0]]
    # print(new_guess)
    # print("---------------")

    # 多源
    # emailW3C 是单源快照，因此默认返回排名第一的节点。
    # 如果运行原来的多源实验，可将上方 SOURCE_COUNT 改成 2，此时返回 tuple。
    top_indices = sorted_indices[:SOURCE_COUNT]
    guess_s = [sorted_nodes[index] for index in top_indices]
    return guess_s[0] if SOURCE_COUNT == 1 else tuple(guess_s)

def get_path():

    paths = list()
    # file_path = 'Algorithm_diagram_snapshot'
    # file_path = 'cps_3_snapshot'
    # file_path = 'hc_2_snapshot'
    # 当前使用配置区中的 emailW3C_snapshot_* 目录。
    # 旧版 sc_2_snapshot 目录由 USE_EMAIL_W3C=False 保留。
    # file_path = 'sb_3_snapshot'
    # file_path = 'WS_uniform_2_snapshot'
    # file_path = 'BA_uniform_snapshot'

    names = os.listdir(SNAPSHOT_DIR)

    # 添加其路径
    for name in names:
        # 只读取快照 CSV，避免目录中的其他文件被误处理。
        if name.lower().endswith('.csv'):
            paths.append(os.path.join(SNAPSHOT_DIR, name))

    # # 添加其路径
    # for name in names:
    #     p = file_path + '/' + name
    #     if re.match(r'ad-2-1-new\.csv', name):
    #         #
    #         paths.append(p)

    return paths

def read_snapshot(g, path_ss):

    # # 计算所有节点的介数中心性
    # betweenness_centrality = nx.betweenness_centrality(g)
    #
    # # 找到具有最大介数中心性的节点
    # # b = max(betweenness_centrality, key=betweenness_centrality.get)
    # # print(b)
    datas = pd.read_csv(path_ss, header=0)
    nums = list()  # 总的感染节点编号列表
    id_obs = dict()  # 监测站节点编号
    obs_pd = dict()  # 监测站收集节点编号
    guess_source = -1
    for i in range(len(datas)):
        temp = datas["id_i"][i]
        nums.append(temp)

    if APPLY_SNAPSHOT_RATIO_IN_ALGORITHM:
        threshold = max(1, int(g.number_of_nodes() * SNAPSHOT_RATIO / 100))
        if len(nums) > threshold:
            nums = random.sample(nums, threshold)
    # 如果nums只有1个节点，那么快照无用，直接获取信息
    count_non_zero = sum(1 for item in nums if item != 0)
    if count_non_zero == 1:
        guess_source = nums[0]
        # print(guess_source)
        # print(1)
        return guess_source, 0, 0, 0
    # 检查到最后一个监测站也为空的标志
    mvalue_null = 0
    # 监测站不记录信息，则无用

    min_obs_time = 10
    # 不可能的最小记录时间
    # 需要删除的键的列表
    keys_to_remove = []
    for i in range(0, 3):
        id_obs[i] = datas["id_obs"][i]
        # 默认等于第一个
        min_obs = datas["obs_pd"][i]
        # 先判断是否监测站全为空
        if datas["obs_time"][i] == -1:
            mvalue_null += 1
            keys_to_remove.append(i)
        # # 记录的绝对时间，即源从0开始传播
        # elif datas["obs_time"][i] == 0:
        #     guess_source = datas["obs_pd"][i]
        #     return guess_source, 0, 0, 0
        # 其他时间
        else:
            if min_obs_time > datas["obs_time"][i]:
                min_obs_time = datas["obs_time"][i]
                min_obs = datas["obs_pd"][i]
            obs = datas["obs_pd"][i]


            # 收集监测站信息
            obs_pd[i] = obs

        if mvalue_null == 3:
            infected_subgraph = g.subgraph(nums)
            id_obs.clear()
            # for node in infected_subgraph.nodes():
                # print(node)
            # print(1)
            return guess_source, infected_subgraph, id_obs, obs_pd
    # 选择监测站获取时间最小的那个
    # for i in range(0, 3):
    #     if datas["obs_time"][i] == min_obs_time:
    #         min_obs = datas["id_obs"][i]
    # 在迭代完成后删除这些键
    for key in keys_to_remove:
        del id_obs[key]

    specified_edges = list()
# 单源
    if VERBOSE:
        print(id_obs, obs_pd)
    # print("min_obs:", min_obs)
    # for neigbor in g.adj[min_obs]:
    #     pair = [min_obs, neigbor]
    #     specified_edges.append(pair)

    # 遍历第一个字典的键和值, 真实网络
    # for key, value in id_obs.items():
    #     if key in obs_pd:
    #         # 键在第二个字典中也存在
    #         # 创建一个包含两个字典中的值的元组并添加到列表中
    #         pair = [value, min_obs]
    #         specified_edges.append(pair)

    # 人造网络
    for key, value in id_obs.items():
        if key in obs_pd:
            # 键在第二个字典中也存在
            # 创建一个包含两个字典中的值的元组并添加到列表中
            pair = [value, obs_pd[key]]
            specified_edges.append(pair)


    # 存储结果的列表
    # print(specified_edges)
    # # 将字典中的所有值赋值给一个集合
    given_nodes = set(obs_pd.values())
    # 找到给定节点之间的所有路径
    # print(id_obs)
    # print(obs_pd)
    # print(given_nodes)
    # 使用 itertools.combinations() 函数输出集合中所有两两组合的元素，并对元素进行操作
    for first, second in combinations(given_nodes, 2):
        # print(f"组合的元素是：{first} 和 {second}")
        # 找到两点之间的最短路径
        distance, path_to_must_pass = shortest_path_hypergraph(first, second, g)
        specified_edges.append(path_to_must_pass)
        # print(path_to_must_pass)

    # 输出结果
    # print("all_paths:", all_paths)
    # 从所有路径中构建感染子图
    infection_subgraph = nx.Graph()
    for path in specified_edges:
        infection_subgraph.add_edges_from(zip(path, path[1:]))



    return guess_source, infection_subgraph, id_obs, obs_pd

def get_real_source(path):
    result = re.search(
        rf'{re.escape(SNAPSHOT_PREFIX)}-(.*?)-',
        os.path.basename(path),
    )
    if result is None:
        raise ValueError(f'Cannot extract source node from snapshot path: {path}')
    return result.group(1)
    # 原来的旧正则代码保留在下面的多行注释中。
    # 当前 emailW3C 文件名格式：emailW3C-1088-0-new.csv。
    if USE_EMAIL_W3C:
        result = re.search(r'emailW3C-(.*?)-', os.path.basename(path))
    else:
        # 旧版 sc 文件名格式：sc-xxx-...。
        result = re.search(r'sc-(.*?)-', os.path.basename(path))

    if result is None:
        raise ValueError(f'无法从快照文件名中提取真实源节点: {path}')
    return result.group(1)

    # 以下是原来的正则写法，保留作对照，不再执行。
    """
    # 当前 emailW3C 文件名格式：emailW3C-1088-0-new.csv。
    if USE_EMAIL_W3C:
        result = re.search(r'emailW3C-(.*?)-', os.path.basename(path))
    else:
        # 旧版 sc 文件名格式：sc-xxx-...。
        result = re.search(r'sc-(.*?)-', os.path.basename(path))

    if result is None:
        raise ValueError(f'无法从快照文件名中提取真实源节点: {path}')
    return result.group(1)
    # 以下是原来的正则写法，保留作对照，不再执行。
    # 获得真实源节点
    # result = re.findall(r"cps_3_snapshot/cps-(.*?)-", path)
    # result = re.findall(r"hc_2_snapshot/hc-(.*?)-", path)
    # result = re.findall(r"sc_2_snapshot/sc-(.*?)-", path)
    # result = re.findall(r"sb_3_snapshot/sb-(.*?)-", path)
    # result = re.findall(r"WS_uniform_2_snapshot/(.*?)-", path)
    # result = re.findall(r"WS_uniform_3_snapshot/WS-(.*?)-", path)
    # print(result[0])
    return result[0]
    """

def get_source(lg, path, obs, pd):
    # real_source = int(get_real_source(path))
    real_source = get_real_source(path)
    # guess_source = rc(nums, g)
    guess_source = get_MSDLP_source(lg, obs, pd)
    # guess_source = get_ibc_source(nums, g)

    if VERBOSE:
        print(real_source, guess_source)
    return real_source, guess_source

# 计算真实源和估计源的误差距离
def shortest_path_hypergraph(start_node, end_node, g):
    if pd.isna(start_node) or pd.isna(end_node):
        return -1, []
    if start_node not in g or end_node not in g:
        return -1, []
    # 超图
    if nx.has_path(g, source=start_node, target=end_node):
        distance = nx.shortest_path_length(g, source=start_node, target=end_node)
        path = nx.shortest_path(g, source=start_node, target=end_node)
    else:
        return -1, []
    # print(distance, path)
    return distance, path

    # if start_node == end_node:
    #     return 0, [start_node]
    #
    #     # BFS 初始化
    # queue = deque([(start_node, 0, [start_node])])
    # visited = set([start_node])
    #
    # while queue:
    #     current_node, distance, path = queue.popleft()
    #
    #     # 遍历当前节点的所有超边
    #     for edge_id, edge in enumerate(H.edges):
    #         for neighbor in H.edges[edge_id]:
    #             if neighbor == end_node:
    #                 return distance + 1, path + [end_node]
    #             if neighbor not in visited:
    #                 visited.add(neighbor)
    #                 queue.append((neighbor, distance + 1, path + [neighbor]))
    #
    # return -1, []  # 如果没有找到路径

# 计算真实源和估计源的误差距离
def get_error_distance(real_s, guess_s, g):
    try:
        # 将 real_s 转换为 int 类型
        real_s = int(real_s)
    except ValueError:
        if VERBOSE:
            print(f"Cannot convert real_s ('{real_s}') to an integer.")
        return None

        # 检查 real_s 和 guess_s 是否在图中
    # 兼容单源节点和旧版多源 tuple；多源时取最小误差距离。
    if isinstance(guess_s, (tuple, list, set)):
        guess_nodes = list(guess_s)
    else:
        guess_nodes = [guess_s]

    valid_nodes = [node for node in guess_nodes if node in g]
    if real_s not in g or not valid_nodes:
        if VERBOSE:
            print(f"Node {real_s} or {guess_s} not in graph.")
        return None

        # 检查是否存在从 real_s 到 guess_s 的路径
    distances = [
        nx.shortest_path_length(g, source=real_s, target=node)
        for node in valid_nodes
        if nx.has_path(g, source=real_s, target=node)
    ]
    if distances:
        distance = min(distances)
        if VERBOSE:
            print(distance)
        return distance

    if VERBOSE:
        print("No path between the nodes.")
    return None

def save_csv(rs, ps, dis):

    # path = r'cps_locating_result_75/MSDLP.csv'
    # path = r'hc_locating_result/MSDLP.csv'
    # path = r'sc_locating_result/MSDLP_ce.csv'
    # path = r'sb_locating_result/MSDLP_ce.csv'
    # path = r'my_idea_result/HLPSI_sample_sb2.csv'
    # 当前输出路径由配置区的 OUTPUT_PATH 决定。
    # 原来的 sc_2_locating_result_50/MSDLP_2.csv 已在配置区保留。
    path = OUTPUT_PATH
    # path = r'D:/python project/wzqProject1/My study/WS_uniform_locating_result/MSDLP_WS_uniform.csv'
    # path = r'WS_uniform_2_locating_result/MSDLP_3.csv'

    # 获取目录路径
    directory = os.path.dirname(path)

    # 如果目录不存在，创建目录
    if not os.path.exists(directory):
        os.makedirs(directory)

    dataframe = pd.DataFrame({
        "real": rs,
        "predict": ps,
        "error_distance": dis
    })
    correct_flags = []
    for real_source, predicted_source in zip(rs, ps):
        try:
            correct_flags.append(int(real_source) == int(predicted_source))
        except (TypeError, ValueError):
            correct_flags.append(real_source == predicted_source)

    total_files = len(dataframe)
    correct_count = sum(correct_flags)
    accuracy = correct_count / total_files if total_files else 0.0
    valid_distances = pd.to_numeric(
        dataframe["error_distance"], errors="coerce"
    ).dropna()
    average_error_distance = (
        valid_distances.mean() if not valid_distances.empty else float("nan")
    )

    dataframe["is_correct"] = correct_flags
    dataframe["accuracy"] = accuracy
    dataframe["average_error_distance"] = average_error_distance
    if VERBOSE:
        print(path)
    dataframe.to_csv(path, index=False, sep=',')

    summary_path = os.path.join(directory, "MSDLP_summary.csv")
    summary = pd.DataFrame([{
        "dataset": DATASET,
        "all_hyperedges_probability": ALL_HYPEREDGES_PROBABILITY,
        "one_hyperedge_probability": 1 - ALL_HYPEREDGES_PROBABILITY,
        "snapshot_ratio": SNAPSHOT_RATIO,
        "total_files": total_files,
        "correct_count": correct_count,
        "accuracy": accuracy,
        "valid_error_distance_count": len(valid_distances),
        "average_error_distance": average_error_distance,
    }])
    summary.to_csv(summary_path, index=False, sep=',')
    return summary.iloc[0].to_dict()

def main():
    G = networks_sir().handle_data()
    paths = get_path()
    rs = list()
    ps = list()
    dis = list()
    # 对文件内容进行获取
    for path in paths:
        if VERBOSE:
            print(path)
        # guess_s, lg, obs_set, pd_set = read_snapshot(H, path)
        real_s = get_real_source(path)
        guess_s, lg, obs_set, pd_set = read_snapshot(G, path)
        # print(real_s, guess_s)
        # 判断监测站记录的绝对时间有无源信息
        if guess_s < 0:
            # 估计源的方法
            real_s, guess_s = get_source(lg, path, obs_set, pd_set)
        # 计算误差距离
        # error_distance, path = shortest_path_hypergraph(real_s, guess_s, H)
        # print(error_distance, path)
        error_distance = get_error_distance(real_s, guess_s, G)

        rs.append(real_s)
        ps.append(guess_s)
        dis.append(error_distance)
    return save_csv(rs, ps, dis)


def run_all_available_experiments():
    """Run MSDLP for every complete parameter snapshot directory on disk."""
    summaries = []
    for dataset in DATASET_CONFIG:
        for all_hyperedges_probability in ALL_EXPERIMENT_PROBABILITIES:
            for snapshot_ratio in ALL_EXPERIMENT_SNAPSHOT_RATIOS:
                configure_experiment(
                    dataset, all_hyperedges_probability, snapshot_ratio
                )
                if not os.path.isdir(SNAPSHOT_DIR):
                    continue

                snapshot_files = [
                    name for name in os.listdir(SNAPSHOT_DIR)
                    if name.lower().endswith(".csv")
                ]
                if len(snapshot_files) != 100:
                    print(
                        f"Skip {SNAPSHOT_DIR}: expected 100 CSV files, "
                        f"found {len(snapshot_files)}."
                    )
                    continue

                print(
                    f"Running {dataset}: "
                    f"{PROBABILITY_LABEL}, snapshot {snapshot_ratio}%"
                )
                summary = main()
                summaries.append(summary)

    if summaries:
        batch_summary_path = os.path.join(
            SCRIPT_DIR, "MSDLP_all_experiments_summary.csv"
        )
        pd.DataFrame(summaries).to_csv(
            batch_summary_path, index=False, sep=","
        )
        print(f"Saved batch summary to {batch_summary_path}")
    else:
        print("No complete parameter snapshot directories were found.")

    return summaries

if __name__ == '__main__':
    if RUN_ALL_AVAILABLE_EXPERIMENTS:
        run_all_available_experiments()
    else:
        main()

