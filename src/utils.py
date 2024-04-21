from itertools import combinations
import string
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
from constants import NEARNESS_SCALE, W, H, ZONE_W, ZONE_H
from skimage.measure import label, regionprops
from matplotlib.colors import ListedColormap
from scipy.spatial.distance import cdist, euclidean, cityblock
import roads.globals as globals
import heapq

def calculate_community_fitness(chromosome: list[int], 
                                buildings_df: pd.DataFrame,
                                building_relationships_df: pd.DataFrame,
                                zone_relationships_df: pd.DataFrame,
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

    building_in_correct_zone_score = calculate_percent_of_buildings_in_correct_zone(chromosome, buildings_df, W, H)
    zone_proximity_score = calculate_overall_zone_proximity_score(chromosome, zone_relationships_df, W, H)
    building_proximity_score = calculate_overall_building_proximity_score(chromosome, building_relationships_df, W, H)
    commute_score = calculate_commute_score(chromosome, W, H)

    score = (0.5 * building_in_correct_zone_score) + \
            (0.25 * zone_proximity_score) + \
            (0.20 * building_proximity_score) + \
            (0.05 * commute_score)
    
    return score



def calculate_percent_of_buildings_in_correct_zone(chromosome: list[int], 
                                                   buildings_df: pd.DataFrame,
                                                   W: int,
                                                   H: int):
    # get the number of buildings in the community
    N = len(buildings_df)    

    # calculate the percent of buildings in the correct zone 
    zone_score = 0
    for i in range(W // ZONE_W):
        for j in range(H // ZONE_H):
            zone_id = chromosome[W * H + i * W // ZONE_W + j]
            if zone_id != 0:
                building_id = chromosome[i * W // ZONE_W + j]
                building = buildings_df[buildings_df["id"] == building_id]
                if building["zone"].values[0] == zone_id:
                    zone_score += 1
                
    zone_score /= N
    score = zone_score
    return score



def calculate_overall_zone_proximity_score(chromosome: list[int],
                                           zone_relationships_df: pd.DataFrame,
                                           W: int,
                                           H: int):
    # Get the important zone relationships
    # ('X' means the two zone types should be far away from each other)
    # ('E' means the two zone types should be near each other)
    relationships_df = zone_relationships_df[(zone_relationships_df["relationship"] == "X") |
                                             (zone_relationships_df["relationship"] == "E")]

    # Calculate the proximity score for each important zone relationship
    score = 0
    for index, row in relationships_df.iterrows():
        if (row["relationship"] == "X"):
            desired_rel_position = "far"
        else:
            desired_rel_position = "near"
        score += calculate_building_proximity_score(chromosome, row["id1"], row["id2"], W, H, desired_rel_position)

    # Take the average of the promixity scores
    score /= relationships_df.shape[0]
    return score



def calculate_zone_proximity_score(chromosome: list[int], 
                                   zone_1: int,
                                   zone_2: int,
                                   W: int,
                                   H: int,
                                   desired_rel_position: string):    
    # Reshape the remaining elements into a W // ZONE_W x H // ZONE_H zone grid
    zones_grid = np.array(chromosome[W*H:]).reshape((W // ZONE_W, H // ZONE_H))

    # Get the zones
    zones = []
    for zone_id in np.unique(zones_grid):
        if zone_id != 0:  # Skip zone 0, which is considered background
            # Find connected components (clusters) of the current zone
            labeled_zones = label(zones_grid == zone_id, connectivity=2)  # Set connectivity to 2
            zones.extend(regionprops(labeled_zones))
        
    # Get all instances of both zones
    zone_1s = []
    zone_2s = []
    zone_label = 0
    for zone in zones:
        if (zone.label == 1):
            zone_label += 1
        if (zone_label == zone_1):
            zone_1s.append(zone.coords)
        elif (zone_label == zone_2):
            zone_2s.append(zone.coords)

    # If both types of zones are not present, 
    # a perfect score is given if "far" is the preferred relative position, and
    # a minimum score is given if "near" is the preferred relative position

    if (len(zone_1s) == 0 or len(zone_2s) == 0):
        if (desired_rel_position == "far"):
            return 1
        elif (desired_rel_position == "near"):
            return 0
    
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
    max_dist = ((W-ZONE_W) ** 2 + (H-ZONE_H) ** 2) ** 0.5

    # Calculate the fitness score
    score = -1
    if (desired_rel_position == "far"):
        score = avg_dist / max_dist
    elif (desired_rel_position == "near"):
        score = (max_dist - avg_dist - 1) / max_dist
    return score



def calculate_overall_building_proximity_score(chromosome: list[int],
                                               building_relationships_df: pd.DataFrame, 
                                               W: int,
                                               H: int):
    # Get the important building relationships
    # ('X' means the two building types should be far away from each other)
    # ('E' means the two building types should be near each other)
    relationships_df = building_relationships_df[(building_relationships_df["relationship"] == "X") |
                                                 (building_relationships_df["relationship"] == "E")]
    
    # Calculate the proximity score for each important building relationship
    score = 0
    for index, row in relationships_df.iterrows():
        if (row["relationship"] == "X"):
            desired_rel_position = "far"
        else:
            desired_rel_position = "near"
        score += calculate_building_proximity_score(chromosome, row["id1"], row["id2"], W, H, desired_rel_position)

    # Take the average of the promixity scores
    score /= relationships_df.shape[0]
    return score
    


def calculate_building_proximity_score(chromosome: list[int], 
                                       building_1: int,
                                       building_2: int,
                                       W: int,
                                       H: int,
                                       desired_rel_position: string):
    # Reshape the first W * H elements of the chromosome into a W x H building grid
    buildings_grid = np.array(chromosome[:W*H]).reshape((W, H))

    # Get the buildings
    buildings = []
    for building_id in np.unique(buildings_grid):
        if building_id != 0:  # Skip building 0, since there is no building 0
            # Find connected components (clusters) of the current building
            labeled_zones = label(buildings_grid == building_id, connectivity=2)  # Set connectivity to 2
            buildings.extend(regionprops(labeled_zones))
        
    # Get all instances of both zones
    building_1s = []
    building_2s = []
    building_label = 0
    for building in buildings:
        if (building.label == 1):
            building_label += 1
        if (building_label == building_1):
            building_1s.append(building.coords)
        elif (building_label == building_2):
            building_2s.append(building.coords)

    # If both types of buildings are not present, 
    # a maximum score is given if "far" is the preferred relative position, and
    # a minimum score is given if "near" is the preferred relative position
    if (len(building_1s) == 0 or len(building_2s) == 0):
        if (desired_rel_position == "far"):
            return 1
        elif (desired_rel_position == "near"):
            return 0
    
    # Calculate the average distance between each building_1 and each building_2
    total_dist = 0
    for z1_coords in building_1s:
        for z2_coords in building_2s:
            test = np.zeros((W, H))
            for x, y in z1_coords:
                test[x][y] = 1
            for x, y in z2_coords:
                test[x][y] = 2
            min_dist = cdist(np.argwhere(test==1),np.argwhere(test==2),'euclidean').min()
            total_dist += min_dist
    avg_dist = total_dist / (len(building_1s) * len(building_2s))

    # Calculate the max possible distance
    max_dist = ((W-1) ** 2 + (H-1) ** 2) ** 0.5

    # Calculate the fitness score
    score = -1
    if (desired_rel_position == "far"):
        score = avg_dist / max_dist
    elif (desired_rel_position == "near"):
        score = (max_dist - avg_dist) / max_dist
    return score


def dijkstra(graph, start):
    dists = {vertex: float('inf') for vertex in graph}
    dists[start] = 0
    priority_queue = [(0, start)]

    while priority_queue:
        current_dist, current_vertex = heapq.heappop(priority_queue)

        if current_dist > dists[current_vertex]:
            continue

        for neighbor in graph[current_vertex]:
            dist = current_dist + graph[current_vertex][neighbor]
            if dist < dists[neighbor]:
                dists[neighbor] = dist
                heapq.heappush(priority_queue, (dist, neighbor))

    return dists

def calculate_commute_score(chromosome: list[int],
                            W: int,
                            H: int):

    # average the distance between residential zones and other zones with job opportunities using the road network

    zones_grid = np.array(chromosome[W*H:]).reshape((W // ZONE_W, H // ZONE_H))

    residential_points = set()
    edges = []
    points = {}

    for point in globals.edges:
        if point in globals.coord_id:
            if zones_grid[int(point.x / 2)][int(point.y / 2)] == 1: # residential zone
                residential_points.add(globals.coord_id[point])
            points[globals.coord_id[point]] = (point.x, point.y)
        
        for edge in globals.edges[point]:
            edges.append((globals.coord_id[point], globals.coord_id[edge]))

    if len(residential_points) == 0:
        return 0

    graph = {point: {} for point in points}

    for edge in edges:
        point1, point2 = edge
        dist = euclidean(points[point1], points[point2])
        graph[point1][point2] = dist
        graph[point2][point1] = dist

    score = 0
    all_distances = []

    for start_point in residential_points:
        dists = dijkstra(graph, start_point)

        for point_id, dist in dists.items():
            if point_id not in residential_points:
                all_distances.append(dist)

    # value closer points more heavily since people tend to work near their homes
    score = np.average(all_distances, weights=np.linspace(1, 0, len(all_distances)))

    # normalize
    max_dist = cdist([[0,0], [W ,0]], [[W, H], [0, H]], 'cityblock').max()
    score = abs(1 - ((max_dist - score) / max_dist))

    return score


    
def plot_solution(chromosome, buildings_df, zones_df, W, H):
    # reshape the first W * H elements of the chromosome into a W x H building grid
    buildings_grid = np.array(chromosome[:W*H]).reshape((W, H))
    
    # reshape the remaining elements into a W // ZONE_W x H // ZONE_H zone grid
    zones_grid = np.array(chromosome[W*H:]).reshape((W // ZONE_W, H // ZONE_H))
    
    # Create a color map based on unique values in the chromosome
    unique_values = np.unique(buildings_grid)
    color_map = plt.cm.get_cmap('viridis', len(unique_values))
    
    # Plot the grid
    plt.figure(figsize=(20, 12))  # Adjust figure size as needed
    plt.imshow(buildings_grid, cmap=color_map, interpolation='nearest')
    
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
    
    print("Zones grid", zones_grid)
    
    # scale zones_grid to size of buildings_grid
    zones_grid = np.repeat(np.repeat(zones_grid, ZONE_W, axis=0), ZONE_H, axis=1)

    # Label zones with bounding boxes
    for zone_id in np.unique(zones_grid):
        if zone_id != 0:  # Skip zone 0, which is considered background
            # Find connected components (clusters) of the current zone
            labeled_zones = label(zones_grid == zone_id, connectivity=2)  # Set connectivity to 2
            regions = regionprops(labeled_zones)
            print("Regions", regions)
            # Draw bounding boxes around each cluster of the current zone
            for region in regions:
                min_y, min_x, max_y, max_x = region.bbox
                width = max_x - min_x
                height = max_y - min_y
                plt.gca().add_patch(plt.Rectangle((min_x - 0.5, min_y - 0.5), width, height, fill=False, edgecolor='red', linewidth=2))
                
                # Get zone type from zones_df based on zone_id
                zone_type = zones_df[zones_df["id"] == zone_id]["type"].values[0]
                
                # Annotate the zone with its type
                plt.text((min_x + .5), (min_y + .5), zone_type, color='black', ha='center', va='center')

    plt.show()
