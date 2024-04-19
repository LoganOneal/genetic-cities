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