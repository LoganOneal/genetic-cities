import numpy as np
import pandas as pd
import pygad
import cvxpy as cp
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time

from dataloaders import load_zone_info, load_zone_relationship_info, load_building_info, load_building_relationship_info, load_relationship_graph
import constants
from initialization import initialize_population
from utils import calculate_community_fitness, plot_solution

zones_df = load_zone_info("./data/zones.csv")
zone_relationships_df = load_zone_relationship_info("./data/zone_relationships.csv")
buildings_df  = load_building_info("./data/buildings.csv")
building_relationships_df = load_building_relationship_info("./data/building_relationships.csv")

#building_relationship_graph, building_must_be_close = load_relationship_graph(num_buildings, "./data/building_relationships.csv")

n_iter = 1
def main():
    def fitness(ga_instance, solution, solution_idx):
            return calculate_community_fitness(solution, buildings_df, building_relationships_df, zone_relationships_df, constants.W, constants.H, return_solution=True)
   
    def on_generation(ga_instance):
        global n_iter
        print("Generation: ", n_iter)
        n_iter += 1

    def mutation_func(offspring, ga_instance):
        for chromosome_idx in range(offspring.shape[0]):
            if np.random.rand() < constants.MUTATION_RATE:
                gene_idx = np.random.randint(0, offspring.shape[1])

                # Separates the chromosome into building and zone genes
                if gene_idx < constants.W * constants.H:
                    offspring[chromosome_idx, gene_idx] = np.random.randint(1, len(buildings_df)+1)
                else:
                    offspring[chromosome_idx, gene_idx] = np.random.randint(1, len(zones_df)+1)
        return offspring

    initial_population = initialize_population(constants.POPULATION_SIZE, buildings_df, zones_df, constants.W, constants.H)

    ga_instance = pygad.GA(num_generations=constants.NUM_GENERATIONS,
                           num_parents_mating=constants.NUM_PARENTS_MATING,
                           fitness_func=fitness,
                           initial_population=initial_population,
                           gene_type=int,
                           gene_space=range(1, len(buildings_df)+1),
                           parent_selection_type="sss",
                           keep_parents=-1,
                           crossover_type="uniform",
                           mutation_type=mutation_func,
                           on_generation=on_generation, 
                          )
    
    start_time = time.time()
    print("Start")
    ga_instance.run()
    finish_time = time.time()
    print("GA Summary", ga_instance.summary())
    print("Best Solution: ", ga_instance.best_solution())
    print("Seconds Elapsed: ", start_time - finish_time)

    best_chromosome, best_score, _ = ga_instance.best_solution()
    print(best_score)

    # plot the best solution
    plot_solution(best_chromosome, buildings_df, zones_df, constants.W, constants.H)

    #ga_instance.plot_fitness()
    # ga_instance.plot_genes()
    # ga_instance.plot_new_solution_rate()


if __name__ == "__main__":
    main()