import json
import hypernetx as hnx

# def load_hypergraph_from_json(file_name='BA_Non-uniform_hypergraph.json'):
def load_hypergraph_from_json(file_name='BA_uniform_hypergraph.json'):
    with open(file_name, 'r') as f:
        hypergraph_data = json.load(f)

    # 解析超边数据
    edges = hypergraph_data.get('edges', {})

    # print(edges)
    # 将超边转换为 Hypergraph 对象
    H = hnx.Hypergraph({int(edge_id): sorted(list(edges[edge])) for edge_id, edge in enumerate(edges)})
    return H

# 从 JSON 文件中加载超图
H = load_hypergraph_from_json()


# # 打印超边信息
# print("超边及其节点编号:")
# for edge_id, edge in enumerate(H.edges):
#     print(f"超边 {edge_id}: {sorted(list(H.edges[edge]))}")
