import pandas as pd
from constants import NEARNESS_SCALE
import networkx as nx

def load_zone_info(file_path):
    zones = pd.read_csv(file_path)

    return zones

def load_zone_relationship_info(file_path):
    zone_relationships_info_df = pd.read_csv(file_path)

    return zone_relationships_info_df  

def load_building_info(file_path):
    buildings_df = pd.read_csv(file_path)

    return buildings_df

def load_building_relationship_info(file_path):
    building_relationships_df = pd.read_csv(file_path)

    return building_relationships_df

def load_relationship_graph(N, file_path):
    relationships_df = pd.read_csv(file_path, header='infer')
    relationships_df["relationship"] = relationships_df["relationship"].map(NEARNESS_SCALE)
    relationships = relationships_df.to_numpy()

    must_be_close = []
    relationship_graph = nx.Graph()
    relationship_graph.add_nodes_from(range(N))
    for (i, j, weight) in relationships:
        if weight == NEARNESS_SCALE["A"]:
            must_be_close.append((i - 1, j - 1))
        relationship_graph.add_edge(i - 1, j - 1, weight=weight)
    
    return relationship_graph, must_be_close