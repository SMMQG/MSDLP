# 均匀超网络，即超边大小固定
import random
import hypernetx as hnx
import networkx as nx
import json

# 均匀超网络，即超边大小固定
def generate_ws_hypergraph(n, k, p, edge_size):
    # 生成一个WS小世界网络
    WG = nx.watts_strogatz_graph(n, k, p)

    # 打印边的数量
    print(f"WS小世界网络中的边数量: {len(WG.edges())}")

    # 初始化超图边
    hyperedges = []

    # 遍历WS图中的边，将其扩展为超边
    for edge in WG.edges():
        u, v = edge
        # 随机选择 edge_size-2 个节点，组成超边
        available_nodes = list(set(WG.nodes()) - {u, v})
        other_nodes = random.sample(available_nodes, edge_size - 2)
        hyperedges.append([u, v] + other_nodes)

    # 确保所有节点都至少在一个超边中
    for node in WG.nodes():
        if not any(node in edge for edge in hyperedges):
            # 随机选择一个超边并添加此节点
            random.choice(hyperedges).append(node)

    # 创建超图
    H = hnx.Hypergraph({i: edge for i, edge in enumerate(hyperedges)})
    return H

# 非均匀网络，超边大小随机，而超边数目确定
def create_small_world_hypergraph(n, k, p, num_hyperedges):
    if k % 2 != 0:
        raise ValueError("k must be an even number")

    nodes = list(range(n))
    edges = []

    # 创建初始的超边
    for _ in range(num_hyperedges):
        edge = set()
        while len(edge) < k:
            edge.add(random.choice(nodes))
        edges.append(edge)

    # 重新连边
    new_edges = []
    for edge in edges:
        new_edge = set(edge)
        for node in list(edge):
            if random.random() < p:
                possible_nodes = list(set(nodes) - new_edge)
                if possible_nodes:
                    new_node = random.choice(possible_nodes)
                    new_edge.remove(node)
                    new_edge.add(new_node)
        new_edges.append(new_edge)

    # 再次检查，确保每个节点都至少在一个超边中
    for node in nodes:
        if not any(node in edge for edge in new_edges):
            edge = random.choice(new_edges)
            edge.add(node)

    H = hnx.Hypergraph({i: list(edge) for i, edge in enumerate(new_edges)})
    return H


def save_hypergraph_to_json(H, file_name='WS_uniform_hypergraph.json'):
    hypergraph_data = {
        'edges': {edge_id: sorted(list(H.edges[edge])) for edge_id, edge in enumerate(H.edges)}
    }

    with open(file_name, 'w') as f:
        json.dump(hypergraph_data, f, indent=4)

# 示例参数
n = 500  # 节点数
k = 4   # 每个超边的节点数
p = 0.1 # 重新连接的概率
# num_hyperedges = 300  # 超边数量
edge_size = 5  # 每个超边的大小（均匀）

# 生成小世界超图网络
H = generate_ws_hypergraph(n, k, p, edge_size)  # 5-均匀超边网络

# H = create_small_world_hypergraph(n, k, p, num_hyperedges)  # 非均匀超边网络

# 保存超图到 JSON 文件
save_hypergraph_to_json(H)
# 打印超图信息
print("超图的超边:")
for edge_id, edge in enumerate(H.edges):
    print(f"超边 {edge_id }: {sorted(list(H.edges[edge]))}")