import numpy as np
import pandas as pd
import pygad
import cvxpy as cp
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time

from dataloaders import load_zone_info, load_building_info, load_relationship_graph
from constants import NEARNESS_SCALE, W, H
from initialization import initialize_population
from utils import calc_dist_based_fitness_score_for_regions, calc_dist_based_fitness_score_within_region, calculate_community_fitness, plot_solution

zones_df = load_zone_info("./data/zones.csv")

buildings_df  = load_building_info("./data/buildings.csv")

#building_relationship_graph, building_must_be_close = load_relationship_graph(num_buildings, "./data/building_relationships.csv")

n_iter = 1
def main():
    def fitness(ga_instance, solution, solution_idx):
            return calculate_community_fitness(solution, zones, buildings_df, zones_df, W, H, return_solution=True)
   
    def on_generation(ga_instance):
        global n_iter
        print("Generation: ", n_iter)
        n_iter += 1

    initial_population, zones = initialize_population(100, buildings_df, zones_df, W, H)

    ga_instance = pygad.GA(num_generations=1,
                           num_parents_mating=100,
                           fitness_func=fitness,
                           initial_population=initial_population,
                           gene_type=int,
                           gene_space=range(1, len(buildings_df)+1),
                           parent_selection_type="sss",
                           keep_parents=-1,
                           crossover_type="single_point",
                           mutation_type="random",
                           mutation_percent_genes=5,
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
    
    # plot the chromosome, a W x H grid where each cell is a building
    plot_solution(best_chromosome, zones, buildings_df, zones_df, W, H)

    #ga_instance.plot_fitness()
    # ga_instance.plot_genes()
    # ga_instance.plot_new_solution_rate()


if __name__ == "__main__":
    main()