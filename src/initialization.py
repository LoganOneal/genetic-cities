import numpy as np
import networkx as nx


def initialize_population(sol_per_pop, N, relationship_graph):  # TODO - Randomly switch directions
    initial_population = []
    for _ in range(sol_per_pop):
        L, B = get_fruchterman_reingold_relative_positionings(relationship_graph)
        chromosome = np.random.choice([0, 2], size=int(N * (N - 1) / 2))
        k = 0
        for i in range(N - 1):
            for j in range(i + 1, N):
                if (chromosome[k] == 0 and L.has_edge(j, i)) or (chromosome[k] == 2 and B.has_edge(j, i)):
                    chromosome[k] += 1
        initial_population.append(chromosome)
    return initial_population


def get_fruchterman_reingold_relative_positionings(G):
    pos = nx.spring_layout(G)

    L = nx.DiGraph()
    L.add_nodes_from(G.nodes)
    B = L.copy()
    for i, j in G.edges():
        if pos[i][0] <= pos[j][0]:
            L.add_edge(i, j)
        else:
            L.add_edge(j, i)
        if pos[i][1] <= pos[j][1]:
            B.add_edge(i, j)
        else:
            B.add_edge(j, i)
    return L, B
