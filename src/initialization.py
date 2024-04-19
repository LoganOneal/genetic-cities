import numpy as np
import networkx as nx
from constants import ZONE_W, ZONE_H

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

        print("Chromosome", chromosome.shape)
        chromosome = chromosome.flatten()
        initial_population.append(chromosome)
        
    
        
    zones = np.zeros((W // ZONE_W, H // ZONE_H))
    # each zone cell represent a ZONE_W x ZONE_H section
    for i in range(zones.shape[0]):
        for j in range(zones.shape[1]):
            zone = zones_df.sample()
            zones[i, j] = zone["id"]

    zones = zones.flatten()
    
    # append zone to each chromosome
    for i in range(len(initial_population)):
        initial_population[i] = np.append(initial_population[i], zones)
                    
    return initial_population            