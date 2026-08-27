import random
from collections import defaultdict
import json

# class Hypergraph:
#     def __init__(self):
#         self.edges = []
#         self.node_to_edges = defaultdict(set)
#
#     def add_edge(self, nodes):
#         edge_id = len(self.edges)
#         self.edges.append(set(nodes))
#         for node in nodes:
#             self.node_to_edges[node].add(edge_id)
#
#     def add_node(self):
#         new_node_id = len(self.node_to_edges)
#         return new_node_id
#
#     def degree(self, node):
#         return len(self.node_to_edges[node])
#
#     def choose_edge(self):
#         edge_weights = [len(edge) for edge in self.edges]
#         total_weight = sum(edge_weights)
#         rand_val = random.uniform(0, total_weight)
#         cumulative_weight = 0
#         for i, weight in enumerate(edge_weights):
#             cumulative_weight += weight
#             if rand_val < cumulative_weight:
#                 return i
#         return len(edge_weights) - 1
#
# def build_ba_hypergraph(initial_nodes, initial_edges, m, total_nodes, total_edges):
#     hypergraph = Hypergraph()
#
#     # 用给定的初始节点和边初始化超图
#     for edge in initial_edges:
#         hypergraph.add_edge(edge)
#
#     # 初始化超边数到达 total_edges
#     while len(hypergraph.edges) < total_edges:
#         # 随机选择一些节点创建新的超边
#         new_edge_nodes = random.sample(list(hypergraph.node_to_edges.keys()), m)
#         hypergraph.add_edge(new_edge_nodes)
#
#     # 添加新节点并连接到现有超边
#     while len(hypergraph.node_to_edges) < total_nodes:
#         new_node = hypergraph.add_node()
#         chosen_edges = set()
#         while len(chosen_edges) < m:
#             edge_id = hypergraph.choose_edge()
#             if edge_id not in chosen_edges:
#                 chosen_edges.add(edge_id)
#
#         for edge_id in chosen_edges:
#             hypergraph.edges[edge_id].add(new_node)
#             hypergraph.node_to_edges[new_node].add(edge_id)
#
#     return hypergraph
#
#
# def save_hypergraph_to_json(hypergraph, file_path):
#     edges_dict = {str(i): list(edge) for i, edge in enumerate(hypergraph.edges)}
#     with open(file_path, 'w') as json_file:
#         json.dump({"edges": edges_dict}, json_file, indent=4)
#
# # 示例用法，生成500个节点和300个超边的超图
# initial_nodes = [0, 1, 2, 3]
# initial_edges = [{0, 1}, {1, 2}, {2, 3}, {3, 0}]
# m = 2
# total_nodes = 500
# total_edges = 300
#
# hypergraph = build_ba_hypergraph(initial_nodes, initial_edges, m, total_nodes, total_edges)
# # print(hypergraph)

# 保存超图为JSON格式

class Hypergraph:
    def __init__(self):
        self.edges = []
        self.node_to_edges = defaultdict(set)
        self.max_edge_size = 5

    def add_edge(self, nodes):
        if len(nodes) <= self.max_edge_size:
            edge_id = len(self.edges)
            self.edges.append(set(nodes))
            for node in nodes:
                self.node_to_edges[node].add(edge_id)
        else:
            raise ValueError("Edge size exceeds the maximum allowed size.")

    def add_node(self):
        new_node_id = len(self.node_to_edges)
        return new_node_id

    def degree(self, node):
        return len(self.node_to_edges[node])

    def choose_edge(self):
        eligible_edges = [i for i, edge in enumerate(self.edges) if len(edge) < self.max_edge_size]
        if not eligible_edges:
            return None
        edge_weights = [len(self.edges[i]) for i in eligible_edges]
        total_weight = sum(edge_weights)
        rand_val = random.uniform(0, total_weight)
        cumulative_weight = 0
        for i, weight in enumerate(edge_weights):
            cumulative_weight += weight
            if rand_val < cumulative_weight:
                return eligible_edges[i]
        return eligible_edges[-1]


def build_5_uniform_ba_hypergraph(initial_nodes, initial_edges, m, total_nodes, total_edges):
    hypergraph = Hypergraph()

    # 用给定的初始节点和边初始化超图
    for edge in initial_edges:
        hypergraph.add_edge(edge)

    # 初始化超边数到达 total_edges
    while len(hypergraph.edges) < total_edges:
        new_edge_nodes = random.sample(list(hypergraph.node_to_edges.keys()), m)
        hypergraph.add_edge(new_edge_nodes)

    # 添加新节点并连接到现有超边
    while len(hypergraph.node_to_edges) < total_nodes:
        new_node = hypergraph.add_node()
        chosen_edges = set()
        attempts = 0
        while len(chosen_edges) < m and attempts < 10 * m:  # 尝试多次以找到符合条件的超边
            edge_id = hypergraph.choose_edge()
            if edge_id is not None and edge_id not in chosen_edges:
                chosen_edges.add(edge_id)
            attempts += 1

        if not chosen_edges:
            # 如果没有找到符合条件的超边，创建一个新的超边
            new_edge_nodes = random.sample(list(hypergraph.node_to_edges.keys()), m - 1) + [new_node]
            hypergraph.add_edge(new_edge_nodes)
        else:
            for edge_id in chosen_edges:
                if len(hypergraph.edges[edge_id]) < hypergraph.max_edge_size:
                    hypergraph.edges[edge_id].add(new_node)
                    hypergraph.node_to_edges[new_node].add(edge_id)

    return hypergraph


def save_hypergraph_to_json(hypergraph, file_path):
    edges_dict = {str(i): list(edge) for i, edge in enumerate(hypergraph.edges)}
    with open(file_path, 'w') as json_file:
        json.dump({"edges": edges_dict}, json_file, indent=4)


# 示例用法，生成500个节点和300个超边的5-均匀BA超图
initial_nodes = [0, 1, 2, 3]
initial_edges = [{0, 1, 2, 3}, {1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}]
m = 2
total_nodes = 500
total_edges = 300

hypergraph = build_5_uniform_ba_hypergraph(initial_nodes, initial_edges, m, total_nodes, total_edges)

file_path = 'BA_uniform_hypergraph.json'
# save_hypergraph_to_json(hypergraph, file_path)

# 输出超图结构
print("超边:")
for i, edge in enumerate(hypergraph.edges):
    print(f"超边 {i}: {edge}")
print("\n节点到超边的映射:")
for node, edges in hypergraph.node_to_edges.items():
    print(f"节点 {node}: {edges}")
