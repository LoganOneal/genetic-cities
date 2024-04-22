import numpy as np
import networkx as nx
from constants import ZONE_W, ZONE_H, MAX_ROAD_NODES
from roads.RoadGenerator import RoadGenerator

DEBUG = True

def initialize_population(sol_per_pop, buildings_df, zones_df, W, H):
    initialize_road_network(W, MAX_ROAD_NODES)
    
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
        
    
    initial_zones = []
    for _ in range(sol_per_pop):
        zones = np.zeros((W // ZONE_W, H // ZONE_H))
        # each zone cell represent a ZONE_W x ZONE_H section
        for i in range(zones.shape[0]):
            for j in range(zones.shape[1]):
                zone = zones_df.sample()
                zones[i, j] = zone["id"]

        zones = zones.flatten()
        initial_zones.append(zones)
    
    # append zone to each chromosome
    for i in range(len(initial_population)):
        initial_population[i] = np.append(initial_population[i], initial_zones[i])
                    
    return initial_population            
def initialize_road_network(W, max_nodes):
    generator = RoadGenerator(W, max_nodes)
    generator.randomize()
    
    if DEBUG: generator.plot()