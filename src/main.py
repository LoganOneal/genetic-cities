import numpy as np
import pandas as pd
import pygad
import cvxpy as cp
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time

from dataloaders import load_zone_info, load_building_info, load_relationship_graph
from constants import NEARNESS_SCALE, W, L
from initialization import initialize_population
from utils import solve_optimal_layout, plot_solution

#zone_types, zone_min_areas, zone_min_widths = load_zone_info("./data/zones.csv")
#zone_min_areas = 0.25 * zone_min_areas
#zone_max_area_weight = 1

building_types, building_min_areas, building_min_widths, required_zone, N  = load_building_info("./data/buildings.csv")
building_man_area = 0.25 * building_min_areas
building_max_area_weight = 1

building_relationship_graph, building_must_be_close = load_relationship_graph(N, "./data/building_relationships.csv")
n_iter = 1

def main():
    def fitness(ga_instance, solution, solution_idx):
            return solve_optimal_layout(solution, N, W, L, building_min_areas, building_min_widths, building_max_area_weight, building_must_be_close)
   
    def on_generation(ga_instance):
        global n_iter
        print("Generation: ", n_iter)
        n_iter += 1

    initial_population = initialize_population(1000, N, building_relationship_graph)
    ga_instance = pygad.GA(num_generations=2,
                           num_parents_mating=100,
                           fitness_func=fitness,
                           initial_population=initial_population,
                           gene_type=int,
                           gene_space=[0, 1, 2, 3],
                           parent_selection_type="sss",
                           keep_parents=-1,
                           crossover_type="single_point",
                           mutation_type="random",
                           mutation_percent_genes=5,
                           on_generation=on_generation)
    start_time = time.time()
    print("Start")
    ga_instance.run()
    finish_time = time.time()
    print("Seconds Elapsed: ", start_time - finish_time)

    best_chromosome, best_score, _ = ga_instance.best_solution()
    print(best_score)
    x, y, w, l = solve_optimal_layout(best_chromosome, N, W, L, building_min_areas, building_min_widths, building_max_area_weight, building_must_be_close, return_solution=True)
    plot_solution(N, x, y, w, l, W, L)

    ga_instance.plot_fitness()
    # ga_instance.plot_genes()
    # ga_instance.plot_new_solution_rate()


if __name__ == "__main__":
    main()