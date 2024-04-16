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
    
    
    return score
    

    
def plot_solution(chromosome, zones, buildings_df, W, H):
    # Reshape the chromosome array into a 2D grid
    grid = np.array(chromosome).reshape((W, H))
    
    # Create a color map based on unique values in the chromosome
    unique_values = np.unique(grid)
    color_map = plt.cm.get_cmap('viridis', len(unique_values))
    
    # Plot the grid
    plt.figure(figsize=(20, 12))  # Adjust figure size as needed
    plt.imshow(grid, cmap=color_map, interpolation='nearest')
    
    # Add grid lines
    plt.grid(True, which='both', color='black', linewidth=0.5)
    
    # Set labels and title
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.title('Chromosome Grid Visualization')
    
    
    print("buildings_df", buildings_df)
    
    # Add legend mapping chromosome index to building type
    legend_labels = {}
    for index, row in buildings_df.iterrows():
        legend_labels[index] = row['type']
    
    print("legend_labels", legend_labels)
    
    legend_handles = []
    for value in unique_values:
        legend_handles.append(plt.Rectangle((0,0),1,1,color=color_map(value)))
    
    plt.legend(legend_handles, [legend_labels[value-1] for value in unique_values], loc='upper left', title='Building Type', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    
    plt.show()
