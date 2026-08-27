
import random
import networkx as nx
import pandas as pd
import os
import matplotlib.pyplot as plt
from pathlib import Path

ALL_HYPEREDGES_PROBABILITIES = (0.3, 0.6, 0.9)
DEFAULT_ALL_HYPEREDGES_PROBABILITY = 0.5
CURRENT_ALL_HYPEREDGES_PROBABILITY = DEFAULT_ALL_HYPEREDGES_PROBABILITY
SNAPSHOT_RATIOS = (0.25, 0.50, 0.75, 1.00)
DATA_FILE = "hyperedges-contact-primary-school.txt"
BASE_DIR = Path(__file__).resolve().parent

# 全局感染概率
alpha = 0.3  # 感染率
beta = 0.2  # 恢复率

class networks_sir:
    def read_file(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        return [list(map(int, line.strip().split(','))) for line in lines]

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
        # Replace with the path to your txt file
        # d = 3,4,3,2
        # file_path = "data/hyperedges-contact-primary-school.txt"
        # file_path = "data/hyperedges-house-committees.txt"
        # file_path = "data/hyperedges-senate-committees.txt"
        file_path = BASE_DIR / "data" / DATA_FILE



        connections = self.read_file(file_path)
        graph = self.create_graph(connections)
        # self.visualize_graph(graph)
        return graph

def updateNodeState_RP(G, node, obs, t):
    infected_node = list()
    for neibor in G.adj[node]:  # 遍历所有邻居用 G.adj[node]
        if G.nodes[neibor]["state"] == "S":
            p1 = random.random()  # 生成一个0到1的随机数
            if p1 < alpha:
                G.nodes[neibor]["state"] = "I"
                infected_node.append(neibor)
                if neibor in obs:
                    G.nodes[neibor]['infection_time'] = t
                    G.nodes[neibor]['source_node'] = node
                    # print(G.nodes[neibor]['source_node'])
            # print(neibor)
            # print(G.nodes[neibor]["state"])
    p2 = random.random()  # 生成一个0到1的随机数
    if p2 < beta:  # beta的概率恢复
        G.nodes[node]["state"] = "R"  # 将节点状态设置成“R”

    return infected_node

# 根据 SIR 模型，更新单一节点的状态 局部
def updateNodeState_CP(G, node, obs, t):
    list1 = select(G, node)  # 筛选指定的权值相同的邻居列表
    infected_node = list()
    for neibor in list1:  # 遍历邻居，选择权值相同的邻居进行感染
        if G.nodes[neibor]["state"] == "S":
            p1 = random.random()  # 生成一个0到1的随机数
            if p1 < alpha:  # beta的概率恢复
                G.nodes[node]["state"] = "I"  # 将节点状态设置成“R”
                G.nodes[neibor]["state"] = "I"
                infected_node.append(neibor)
                if neibor in obs:
                    G.nodes[neibor]['infection_time'] = t
                    G.nodes[neibor]['source_node'] = node
                    # print(G.nodes[neibor]['source_node'])
            # print(neibor)
            # print(G.nodes[node]["state"])
    p2 = random.random()  # 生成一个0到1的随机数
    if p2 < beta:  # beta的概率恢复
        G.nodes[node]["state"] = "R"  # 将节点状态设置成“R”

    return infected_node

def count_weight(G, node):
    list1 = list()
    for neibor in G.adj[node]:
        a1 = G.get_edge_data(node, neibor)  # 得到每条边的属性，'weight'：值
        b1 = list(a1.values())  # 取出属性的值
        list1.append(b1[0])  # 添加所有权值属性到list2列表
        weight = list(set(list1))  # 将所有权值放在一起，并将重复的去除。
    return weight

# 作为局部策略中，筛选超边进行传播
def select(G, node):
   weight = count_weight(G, node)
   s = random.choice(weight)  # 随机选择一个权值
   list2 = []
   for neibor in G.adj[node]:
       a2 = G.get_edge_data(node, neibor)  # 得到每条边的属性，'weight'：值
       b2 = list(a2.values())  # 取出属性的值
       x = b2[0]  # 转换为int类型
       if s == x:  # 如果选择相同即先在这个权值内的超边进行传播
           list2.append(neibor)
   return list2

# 根据真实源节点，进行传播，并更新节点状态
def updateNetworkState(
    G, s, obs, t,
    all_hyperedges_probability=None,
):

    if all_hyperedges_probability is None:
        all_hyperedges_probability = CURRENT_ALL_HYPEREDGES_PROBABILITY

    if not 0 <= all_hyperedges_probability <= 1:
        raise ValueError("all_hyperedges_probability must be between 0 and 1")

    nums = s
    # print(nums)
    # 存储感染节点编号的列表
    infected_nodes = list()
    for node in nums:
        # print(node)
        # print(G.nodes[node]['state'])
        # 遍历图中节点，每一个节点状态进行更新
        if G.nodes[node]['state'] == "I":
            # 使用 random.choice 来从 infection_choices 中随机选择一种感染方式
            if random.random() < all_hyperedges_probability:
                temp = updateNodeState_RP(G, node, obs, t)  # 全局策略进行传播
            else:
                temp = updateNodeState_CP(G, node, obs, t)  # 局部策略进行传播
            # or G.nodes[node]['state'] == "R":
            infected_nodes += temp
            infected_nodes.append(node)
            # 传播完成后将节点状态设置为 "R"
            # G.nodes[node]['state'] = "R"
            # if i == len(nums) - 1:
            #     break
    nums_0 = list(set(infected_nodes))
    # print(nums)
    return nums_0

def init_state(g, source):
    if isinstance(source, int):
        source = [source]
    # 初始化节点状态
    for node in g.nodes():
        # print(node)
        g.nodes[node]["state"] = "S"
        # print(g.nodes[node])

    # # # 随机选取一个节点为初始感染者
    # r = random.randint(1, len(g.nodes))
    for s in source:
        # print(s)
        g.nodes[s]["state"] = "I"
    # print(g.nodes[1])


def select_observers(g):

    # degree_centrality = nx.degree_centrality(g)  # 计算度中心性
    # # 选择度中心性最大的几个节点作为观察者
    # num_observers = 3
    observers = [101, 10, 157]
    # observers = [412, 530, 183]
    # observers = [14, 112, 18]
    # observers = [131, 99, 283]
    # observers = [21, 3, 13]  # 算法框架网络拓扑

    # 设置观察者节点的属性
    for obs in observers:
        g.nodes[obs]['infection_time'] = -1
        g.nodes[obs]['source_node'] = -1

    return observers

# 获取传播某一个过程的静态数据
def read_snapshot(
    g, source,
    all_hyperedges_probability=DEFAULT_ALL_HYPEREDGES_PROBABILITY,
):
    global CURRENT_ALL_HYPEREDGES_PROBABILITY
    CURRENT_ALL_HYPEREDGES_PROBABILITY = all_hyperedges_probability
    source = (source,) if isinstance(source, int) else tuple(source)
    # days = random.randint(0, 2)  # 设置模拟天数

    init_state(g, source)
    real_source = source  # 获得初始状态的源节点编号

    obs = select_observers(g)
    # nums = g.nodes()
    nums = real_source
    for t in range(0, 10):  # 直径是网络中任意两节点之间最短路径的最大值。
        temp = updateNetworkState(g, nums, obs, t)  # 返回该网络状态下，感染节点和恢复节点的编号。
        nums_list = list(nums)
        nums_list.extend(list(set(temp)))
        nums = tuple(nums_list)
        # 将列表`temp`转换为集合，去除重复的元素，并将结果转换回列表
    # print(real_source)

    nums_0 = list(set(nums))
    return nums_0, real_source, obs

def random_subset(lst, threshold):
    if len(lst) > threshold:
        subset_size = threshold
        selected_elements = random.sample(lst, subset_size)
        return selected_elements
    else:
        return lst

def save_csv(nums, obs, G, path, snapshot_ratio):
    # # # 数据集1
    # # # 生成当前迭代的文件名
    # # current_filename = f'cps-121-{fm}-new.csv'
    # # save_path = 'cps_snapshot'
    # # # 拼接完整的文件路径
    # # path = os.path.join(save_path, current_filename)
    # # 数据集2
    # # # 生成当前迭代的文件名
    # # current_filename = f'hc-48-{fm}-new.csv'
    # # save_path = 'hc_snapshot'
    # # # 拼接完整的文件路径
    # # path = os.path.join(save_path, current_filename)
    #
    # # 数据集3
    # # 生成当前迭代的文件名
    # # current_filename = f'sc-115-{fm}-new.csv'
    # # save_path = 'sc_snapshot'
    # # # 拼接完整的文件路径
    # # path = os.path.join(save_path, current_filename)
    #
    # # 数据集4
    # current_filename = f'sb-16-{fm}-new.csv'
    # save_path = 'sb_snapshot'
    # # 拼接完整的文件路径
    # path = os.path.join(save_path, current_filename)

    # 获取新的列表
    if not 0 < snapshot_ratio <= 1:
        raise ValueError("snapshot_ratio must be greater than 0 and at most 1")

    threshold_value = max(1, int(G.number_of_nodes() * snapshot_ratio))
    new_list = random_subset(nums, threshold_value)

    time = list()
    prior_node = list()
    for i in obs:
        time.append(G.nodes[i]['infection_time'])
        prior_node.append(G.nodes[i]['source_node'])
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

def count_nums_percent(g, nums):
    num_nodes = g.number_of_nodes()
    infected_percent = round(len(nums) / num_nodes, 2)
    if infected_percent == 0:
        return 1
    else:
        print(infected_percent)

def simulate_snapshot(graph, item, all_hyperedges_probability):
    """Run one propagation and return data reusable by all snapshot ratios."""
    nums, real_source, observers = read_snapshot(
        graph, item, all_hyperedges_probability
    )
    return nums, observers

def main(
    path, item,
    all_hyperedges_probability=DEFAULT_ALL_HYPEREDGES_PROBABILITY,
    snapshot_ratio=1.0,
    graph=None,
):
    # 获取高阶网络数据集，并将其转化为一般的网络进行流动传播

    if graph is None:
        graph = networks_sir().handle_data()
    G = graph
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

    # 给定一个快照，获取感染节点列表和真实源节点。
    nums, obs = simulate_snapshot(G, item, all_hyperedges_probability)

    # count_nums_percent(G, nums)

    # print(nums, real_s, obs)

    save_csv(nums, obs, G, path, snapshot_ratio)

if __name__ == "__main__":
    # The same 10 single-source nodes used by the previous 0.5/0.5
    # single-source experiment.
    rand_source = [1, 8, 9, 24, 31, 45, 55, 66, 81, 121]

    # The graph structure is identical across all experiments.  Build it once
    # and reset only node states for each simulation.
    graph = networks_sir().handle_data()

    # rand_source = [(118, 5, 49), (56, 5, 49)]
    for item in rand_source:
        print(item)
        for all_hyperedges_probability in ALL_HYPEREDGES_PROBABILITIES:
            one_hyperedge_probability = 1 - all_hyperedges_probability
            for i in range(10):
                # One propagation is sufficient for all four snapshot ratios.
                # The ratios only change how many infected nodes are sampled.
                nums, obs = simulate_snapshot(
                    graph, item, all_hyperedges_probability
                )
                for snapshot_ratio in SNAPSHOT_RATIOS:
                    ratio_label = int(snapshot_ratio * 100)
                    output_dir = (
                        f'cps_snapshot_all{all_hyperedges_probability:.1f}_'
                        f'one{one_hyperedge_probability:.1f}_{ratio_label}'
                    )
                    path = BASE_DIR / output_dir / f'cps-{item}-{i}-new.csv'
                    save_csv(nums, obs, graph, path, snapshot_ratio)
