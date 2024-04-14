import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
from constants import NEARNESS_SCALE, W, H


def calculate_community_fitness(chromosome: list[int], 
                         zones: list[int],
                         buildings_df: pd.DataFrame,
                         zones_df: pd.DataFrame,
                         W: int,
                         H: int,
                         return_solution: bool=False) -> float:
    """
    Basis for the fitness function for a specific chromosome.
  
    Score a chromosome based on the weighted sum of stakeholder objectives.
  
    Parameters:
    chromosome: list of integers of length W * H + 1, defining the relative positioning of the buildings and the type of road to use
    N: the number of buildings in the community
  
    Returns:
    float: the score of the community layout
    """

    # get the number of buildings in the community
    N = len(buildings_df)    

    # calculate the percent of buildings in the correct zone 
    zone_score = 0
    for i in range(W):
        for j in range(H):
            building_id = chromosome[i * W + j]
            zone_id = zones[i * W + j]
            if zone_id != 0:
                building = buildings_df[buildings_df["id"] == building_id]
                if building["zone"].values[0] == zone_id:
                    zone_score += 1
                
    zone_score /= N
    
    
    score = zone_score
    
    print("Zone Score: ", zone_score)
    
    return score
    

    
def plot_solution(N, x, y, w, l, W, L):
    fig, ax = plt.subplots()
    plt.scatter(x, y, s=0.1, color="black")
    ax.add_patch(Rectangle((0, 0), W, L, edgecolor="black", linestyle="dashed", fill=False))
    for i in range(N):
        ax.add_patch(Rectangle((x[i], y[i]), w[i], l[i], edgecolor="tab:blue", fill=False))
        plt.text(x[i] + (w[i] / 2) - 1, y[i] + (l[i] / 2) - 1, s=str(i))
    plt.show()
    #plt.savefig("./images/solution.pdf")
