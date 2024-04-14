import numpy as np
import networkx as nx
from constants import ZONE_W, ZONE_H

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

def initialize_population(sol_per_pop, buildings_df, zones_df, W, H):
    print("Initializing population")
    # create a population where each chronosome is a W * H grid of buildings with random positions
    min_area = W*H
    if min_area < len(buildings_df):
        raise ValueError("The area of the community is too small for the number of buildings")

    initial_population = []
    for _ in range(sol_per_pop):
        chromosome = np.zeros((W, H))
        # randomly place buildings inside the chromosome
        for i in range(W):
            for j in range(H):
                # get random building 
                building = buildings_df.sample()
                # set chromosme indx to building id
                chromosome[i, j] = building["id"]

        chromosome = chromosome.flatten()
        initial_population.append(chromosome)
        
    zones = np.zeros((W, H))
    # cluster zones together into 2x2 sections
    for i in range(W // ZONE_W):
        for j in range(H // ZONE_H):
            # get random zone
            zone = zones_df.sample()
            # set the section to the zone id 
            zones[i * ZONE_W:(i + 1) * ZONE_W, j * ZONE_H:(j + 1) * ZONE_H] = zone["id"] 
    zones = zones.flatten()
                    
    return initial_population, zones            
        

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
