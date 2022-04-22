import networkx as nx
import pickle
import pandas as pd


if __name__ == "__main__":
    with open("node_with_embedding.csv", "r") as f:
        nodes = pd.read_csv(f)
    with open("edge.csv", "r") as f:
        edges = pd.read_csv(f)
    
    G = nx.Graph()
    node_list = []
    edge_list = []

    for _, row in nodes.iterrows():
        temp_node = list(row)
        node_list.append( (temp_node[1], {"text": temp_node[2], "embedding": temp_node[3]}) )
    G.add_nodes_from(node_list)

    for _, row in edges.iterrows():
        edge_list.append(tuple(row))
    G.add_edges_from(edge_list)

    with open("graph.pickle", "wb") as f:
        pickle.dump(G, f)
