from itertools import combinations
import string
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
from constants import NEARNESS_SCALE, W, H
from skimage.measure import label, regionprops
from matplotlib.colors import ListedColormap
from scipy.spatial.distance import cdist
import roads.globals as globals

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
    


def calc_dist_based_fitness_score_for_regions(chromosome: list[int], 
                                              zones: list[int],
                                              zone_1: int,
                                              zone_2: int,
                                              W: int,
                                              H: int,
                                              desired_rel_position: string):
    # Get the zones
    zones = np.array(zones).reshape((W, H))

    # Get the regions
    regions = []
    for zone_id in np.unique(zones):
        if zone_id != 0:  # Skip zone 0, which is considered background
            # Find connected components (clusters) of the current zone
            labeled_zones = label(zones == zone_id, connectivity=2)  # Set connectivity to 2
            regions.extend(regionprops(labeled_zones))
        
    # Get all instances of both zones
    zone_1s = []
    zone_2s = []
    zone_label = 0
    for region in regions:
        if (region.label == 1):
            zone_label += 1
        if (zone_label == zone_1):
            zone_1s.append(region.coords)
        elif (zone_label == zone_2):
            zone_2s.append(region.coords)

    # If both types of regions are not present, 
    # a perfect score is given if "far" is the preferred relative position
    if (len(zone_1s) == 0 or len(zone_2s) == 0):
        if (desired_rel_position == "far"):
            return 1
    
    # Calculate the average distance between each zone_1 and each zone_2
    total_dist = 0
    for z1_coords in zone_1s:
        for z2_coords in zone_2s:
            test = np.zeros((W, H))
            for x, y in z1_coords:
                test[x][y] = 1
            for x, y in z2_coords:
                test[x][y] = 2
            min_dist = cdist(np.argwhere(test==1),np.argwhere(test==2),'euclidean').min()
            total_dist += min_dist
    avg_dist = total_dist / (len(zone_1s) * len(zone_2s))

    # Calculate the max possible distance
    max_dist = ((W-2) ** 2 + (H-2) ** 2) ** 0.5

    # Calculate the fitness score
    score = -1
    if (desired_rel_position == "far"):
        score = avg_dist / max_dist
    elif (desired_rel_position == "near"):
        score = (max_dist - avg_dist) / max_dist
    return score



def calc_dist_based_fitness_score_within_region(chromosome: list[int], 
                                                zones: list[int],
                                                building_1: int,
                                                building_2: int,
                                                W: int,
                                                H: int,
                                                desired_rel_position: string):
    
    # Get the buildings and zones
    grid = np.array(chromosome).reshape((W, H))
    zones = np.array(zones).reshape((W, H))

    # Get the regions
    regions = []
    for zone_id in np.unique(zones):
        if zone_id != 0:  # Skip zone 0, which is considered background
            # Find connected components (clusters) of the current zone
            labeled_zones = label(zones == zone_id, connectivity=2)  # Set connectivity to 2
            regions.extend(regionprops(labeled_zones))
    
    overall_score = 0
    for region in regions:
        # Get all instancecs of both building types in a given region
        building_1s = []
        building_2s = []
        for x, y in region.coords:
            if (grid[x][y] == building_1):
                building_1s.append([x, y])
            elif (grid[x][y] == building_2):
                building_2s.append([x, y])

        # If both types of buildings are not in the region, 
        # a perfect score is given if "far" is the preferred relative position
        if (len(building_1s) == 0 or len(building_2s) == 0):
            if (desired_rel_position == "far"):
                overall_score += 1
            continue
        
        # Otherwise, calculate the region's fitness score
        # Calculate the max distance for the given region
        max_dist = 0
        for pair in combinations(region.coords, 2):
            x1, y1 = pair[0]
            x2, y2 = pair[1]
            total_dist = ((x2-x1) ** 2 + (y2-y1) ** 2) ** 0.5
            if (total_dist > max_dist):
                max_dist = total_dist
        
        # Calculate the distance between each building 1 and each building 2
        total_dist = 0
        for x1, y1 in building_1s:
            for x2, y2 in building_2s:
                total_dist += ((x2-x1) ** 2 + (y2-y1) ** 2) ** 0.5
        avg_dist = total_dist / (len(building_1s) * len(building_2s))
        if (desired_rel_position == "far"):
            overall_score += (avg_dist / max_dist)
        elif (desired_rel_position == "near"):
            overall_score += ((max_dist - avg_dist) / max_dist)
    
    # Calculate the overall average score across all regions
    overall_avg_score = overall_score / len(regions)
    return overall_avg_score


    
def plot_solution(chromosome, zones, buildings_df, zones_df, W, H):
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
        
    # Add legend mapping chromosome index to building type
    legend_labels = {}
    for index, row in buildings_df.iterrows():
        legend_labels[index] = row['type']
        
    legend_handles = []
    for value in unique_values:
        legend_handles.append(plt.Rectangle((0,0),1,1,color=color_map(value)))
    
    plt.legend(legend_handles, [legend_labels[value-1] for value in unique_values], loc='upper left', title='Building Type', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    
    
    print("Zones df", zones_df)
    
    # convert zones to a 2d W * H grid 
    zones = np.array(zones).reshape((W, H))

    print("Zones", zones)

    # Label zones with bounding boxes
    for zone_id in np.unique(zones):
        if zone_id != 0:  # Skip zone 0, which is considered background
            # Find connected components (clusters) of the current zone
            labeled_zones = label(zones == zone_id, connectivity=2)  # Set connectivity to 2
            regions = regionprops(labeled_zones)
            
            # Draw bounding boxes around each cluster of the current zone
            for region in regions:
                min_y, min_x, max_y, max_x = region.bbox
                width = max_x - min_x
                height = max_y - min_y
                plt.gca().add_patch(plt.Rectangle((min_x - 0.5, min_y - 0.5), width, height, fill=False, edgecolor='red', linewidth=2))
                
                # Get zone type from zones_df based on zone_id
                zone_type = zones_df[zones_df["id"] == zone_id]["type"].values[0]
                
                # Annotate the zone with its type
                plt.text((min_x), (min_y), zone_type, color='black', ha='center', va='center')

    plt.show()
